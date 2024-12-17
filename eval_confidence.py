# -*- coding: utf-8 -*-
import os
import re
import time
import json
import copy
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmtuner.data.template import get_template_and_fix_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/data/oss_bucket_0/huanghui/Meta-Llama-3-8B-Instruct",
)
parser.add_argument(
    "--input-file",
    type=str,
    default=None,
)
parser.add_argument(
    "--output-file",
    type=str,
    default=None,
)
parser.add_argument(
    "--template-type",
    type=str,
    default="llama",
    choices=("llama3", "qwen", "mistral")
)
args = parser.parse_args()

@torch.inference_mode()
def get_batch_evaluation(
    model,
    input_ids,
    total_lens,
    prefix_lens,
):
    """
    Inputs:
        output_ids: The predicted ids consist of both instruction and response, shape is [1, sequence_len]
        prefix_len: The length of the instruction part
    Outputs:
        logprobs: Token level logprobs, shape is [sequence_len]
        entropy: Token level logprobs_entroppy, shape is [sequence_len]
        prefix_len: The length of the instruction part
    """
    input_ids = input_ids.to(model.device)
    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
        output_hidden_states=True,
    )
    shifted_input_ids = torch.roll(input_ids, shifts=-1) # the predict ids should be shifted left
    logits = outputs["logits"]
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    # The original entropy has a minus sign, but we remove it to keep the positive correlation,
    # which means the bigger the logprobs_entropy is, the better the quality is
    # 疑问：这里究竟要不要对logits重新取一遍softmax？
    logprobs_entropy = torch.mean(logprobs * torch.nn.functional.softmax(logits, dim=-1), dim=-1)

    # the bigger the perplexity is, the better the quality is
    perplexity = torch.gather(logprobs, dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)

    logprobs_entropy = logprobs_entropy.squeeze(1).cpu().tolist()
    # logprobs = logprobs.cpu().tolist()
    perplexity = perplexity.squeeze(1).cpu().tolist()

    evaluations = []
    for i in range(len(total_lens)):
        prefix_len = prefix_lens[i]
        total_len = total_lens[i]
        evaluations.append({
            "logprobs": perplexity[i][prefix_len:total_len], 
            "entropy": logprobs_entropy[i][prefix_len:total_len],
            "logprobs_dist": logprobs[i][prefix_len:total_len],
        })

    return evaluations

def evaluate_response(batched_lines, tokenizer, template, model, instruction_key, response_key):

    if args.template_type == "llama":
        # llama3 got some problem for padding
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = []
    total_lens = []
    prefix_lens = []
    for line in batched_lines:

        assert len(line[response_key]) == 1
        response = line[response_key][0]

        if instruction_key == "":
            instruction = ""
        else:
            instruction = line[instruction_key]
        messages = [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]
        prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages, system=None, tools=None)

        total_lens.append(len(prompt_ids + response_ids))
        prefix_lens.append(len(prompt_ids))

        input_ids.append(prompt_ids + response_ids)

    # manually pad the batch
    longest_length = max([len(i) for i in input_ids])
    input_ids = [i + [tokenizer.pad_token_id] * (longest_length-len(i)) for i in input_ids]

    input_ids = torch.LongTensor(input_ids)
    evaluation = get_batch_evaluation(model=model,
                                      input_ids=input_ids,
                                      total_lens=total_lens,
                                      prefix_lens=prefix_lens)
    eval_lines = []
    for prefix_len, eval_line in zip(prefix_lens, evaluation):
        eval_line["source_token_scores"] = [0.0] * prefix_len
        eval_lines.append(eval_line)

    return eval_lines

def cal_confidence_metric(line, line_rev, line_empty):
    kldiv = torch.nn.functional.kl_div(line['logprobs_dist'], line_rev['logprobs_dist'], reduction='none', log_target=True).sum(-1)
    kldiv_rev = torch.nn.functional.kl_div(line_rev['logprobs_dist'], line['logprobs_dist'], reduction='none', log_target=True).sum(-1)

    # 参考：https://github.com/IINemo/lm-polygraph/blob/main/src/lm_polygraph/estimators/pointwise_mutual_information.py
    pmi = [l[0] - l[1] for l in zip(line['logprobs'],line_empty['logprobs'])] 
    pmi_rev = [l[0] - l[1] for l in zip(line_rev['logprobs'],line_empty['logprobs'])] 

    bi_kldiv = kldiv + kldiv_rev

    evaluation = {
        "entropy": line["entropy"],
        "logprobs": line["logprobs"],
        "entropy_rev": line_rev["entropy"],
        "logprobs_rev": line_rev["logprobs"],
        "kldiv": kldiv.cpu().tolist(),
        "bi_kldiv": bi_kldiv.cpu().tolist(),
        "pmi": pmi,
        "pmi_rev": pmi_rev,
    }
    return evaluation

if __name__ == "__main__":
    
    fin = open(args.input_file, "r")
    lines = [json.loads(line) for line in fin.readlines()]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    template = get_template_and_fix_tokenizer(tokenizer, args.template_type)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True).half().cuda()

    pbar = tqdm(total=len(lines))
    bs = 2
    for i in range(0, len(lines), bs):
        batched_lines = lines[i: i+bs]
        eval_lines = evaluate_response(batched_lines, tokenizer=tokenizer, template=template, model=model, instruction_key="instruction", response_key="responses")
        noised_eval_lines = evaluate_response(batched_lines, tokenizer=tokenizer, template=template, model=model, instruction_key="instruction", response_key="noised_responses")
        eval_lines_rev = evaluate_response(batched_lines, tokenizer=tokenizer, template=template, model=model, instruction_key="noised_instruction", response_key="responses")
        noised_eval_lines_rev = evaluate_response(batched_lines, tokenizer=tokenizer, template=template, model=model, instruction_key="noised_instruction", response_key="noised_responses")
        eval_lines_empty = evaluate_response(batched_lines, tokenizer=tokenizer, template=template, model=model, instruction_key="", response_key="responses")
        noised_eval_lines_empty = evaluate_response(batched_lines, tokenizer=tokenizer, template=template, model=model, instruction_key="", response_key="noised_responses")

        for j, line in enumerate(batched_lines):
            line['source_token_scores'] = eval_lines[j]["source_token_scores"]
            line["evaluations"] = cal_confidence_metric(eval_lines[j], eval_lines_rev[j], eval_lines_empty[j])
            line["noised_evaluations"] = cal_confidence_metric(noised_eval_lines[j], noised_eval_lines_rev[j], noised_eval_lines_empty[j])
        
        pbar.update(bs)

    sample_lines = random.sample(lines, k=1)
    for i in range(len(sample_lines)):
        sample_line = sample_lines[i]
        print(f"*************Sampled Score {i}*************")
        print("entropy:" + str(sample_line["evaluations"]['entropy']))
        print("logprobs:" + str(sample_line["evaluations"]['logprobs']))
        print("kldiv:" + str(sample_line["evaluations"]['kldiv']))
        print("pmi:" + str(sample_line["evaluations"]['pmi']))

    with open(args.output_file, "w") as fout:
        for line in lines:
            fout.write(json.dumps(line) + "\n")