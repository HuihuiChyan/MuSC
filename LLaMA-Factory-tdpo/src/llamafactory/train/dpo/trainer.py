# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


def get_batch_logps(
    logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = IGNORE_INDEX
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    r"""
    Computes the log probabilities of the given labels under the given logits.

    Returns:
        logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask), loss_mask


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def get_batch_samples(self, epoch_iterator, num_batches):
        r"""
        Replaces the method of KTO Trainer with the one of the standard Trainer.
        """
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def tdpo_loss(
        self, 
        policy_chosen_logps: "torch.Tensor", 
        policy_rejected_logps: "torch.Tensor", 
        reference_chosen_logps: "torch.Tensor", 
        reference_rejected_logps: "torch.Tensor", 
        chosen_token_scores: "torch.Tensor", 
        rejected_token_scores: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Adapted from https://github.com/huggingface/trl/blob/453db5cd791338d860ea6e8581c853e15b0ae702/trl/trainer/dpo_trainer.py#L874
        Compute the TDPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size, )
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            chosen_token_scores: Reward scores of each token of the chosen response. Shape: (batch_size, sequence_length)
            rejected_token_scores: Reward scores of each token of the rejected response. Shape: (batch_size, sequence_length)
        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the FGMO-DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_rewards = policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device) # bs/2 x (l-1)
        rejected_rewards = policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(self.accelerator.device)

        chosen_rewards = chosen_token_scores * self.beta * chosen_rewards # bs/2 x (l-1)
        rejected_rewards = rejected_token_scores * self.beta * rejected_rewards

        logits = (chosen_rewards - rejected_rewards).sum(-1) # bs/2
        losses = -F.logsigmoid(logits)

        return losses, chosen_rewards.detach().sum(-1), rejected_rewards.detach().sum(-1)

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
        chosen_token_scores: Optional["torch.Tensor"],
        rejected_token_scores: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            if self.loss_type == "tdpo":
                losses, chosen_rewards, rejected_rewards = self.tdpo_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, chosen_token_scores, rejected_token_scores
                )
            
            else:
                losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
                )

        return losses, chosen_rewards, rejected_rewards

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length.unsqueeze(1)

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        forward_batch = {"input_ids": batch["input_ids"], "labels": batch["labels"], "attention_mask": batch["attention_mask"]}
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, forward_batch)

        bs = batch['input_ids'].shape[0] // 2
        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, forward_batch)
        if self.loss_type == "tdpo":
            chosen_token_scores, rejected_token_scores = batch['token_scores'][:bs], batch['token_scores'][bs:]
            losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                chosen_token_scores,
                rejected_token_scores,
            )
        else:
            losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                None, None,
            )            
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

        return losses.mean(), metrics

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Fixes the loss value for transformers 4.46.0.
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        loss = super().compute_loss(model, inputs, return_outputs)
        if is_transformers_version_equal_to_4_46() and kwargs.pop("num_items_in_batch", False):
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        return loss

    @override
    def log(self, logs: Dict[str, float]) -> None:
        r"""
        Log `logs` on the various objects watching training, including stored metrics.
        """
        # logs either has "loss" or "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad to for all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):  # add remaining items
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs)