# MuSC

This repository is for Musc: Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training


# ⚡️ Usage

## Preparation

For pre-existing complex queries, we use the dataset of [WizardLM70k](https://huggingface.co/datasets/cognitivecomputations/wizard_vicuna_70k_unfiltered), and we randomly select 2000 samples.

For self-generate complex queries, we random select in-context demonstrations from [InfoBench](https://huggingface.co/datasets/kqsong/InFoBench), as presented in `data` directory.

## Constraint-aware Preference Data

For pre-existing complex queries, run the following script to decompose instructions into cosntraints.
```shell
i=$1
export CUDA_VISIBLE_DEVICES=$((i))

CKPT=/path/to/Meta-Llama-3-8B-Instruct
DATA=wizard70k-llama3
DATA_DIR=data/llama3

python detach_instruction.py \
    --checkpoint $CKPT \
    --drop-ratio 0.3 \
    --start-idx 0 \
    --response-num 1 \
    --generate-negative True \
    --total-num 1000 \
    --random-seed $i \
    --input-file ./wizard_vicuna_70k_unfiltered/wizard_vicuna_dataset_unfiltered.json \
    --output-file ./$DATA_DIR/$DATA-$i.jsonl
```

For self-instructed complex queries, run the following script to generate instructions based on in-context examples.
```shell
i=$1
export CUDA_VISIBLE_DEVICES=$((i))

CKPT=/path/to/Meta-Llama-3-8B-Instruct
DATA=selfinst-llama3
DATA_DIR=data/llama3

python create_instruction.py \
    --random-seed $i \
    --response-num 1 \
    --generate-negative True \
    --checkpoint $CKPT \
    --total-num 1000 \
    --output-file ./$DATA_DIR/$DATA-$i.jsonl
```

## Confidence-based Token Supervision

Run the following script to derive confidence supervision for token-level DPO.
```shell
i=$1
export CUDA_VISIBLE_DEVICES=$((i))

CKPT=/path/to/Meta-Llama-3-8B-Instruct
DATA_NAME=selfinst-llama3
DATA_DIR=data/llama3

python eval_confidence.py \
    --template-type llama3 \
    --checkpoint $CKPT \
    --input-file "./$DATA_DIR/$DATA_NAME-$i.jsonl" \
    --output-file "./$DATA_DIR/$DATA_NAME-conf-$i.jsonl"
```

## Self-Reward

Run the following script to derive self-rewarding supervision with Branch-Solve-Merge.
```shell
i=$1
export CUDA_VISIBLE_DEVICES=$((i))

CKPT=/path/to/Meta-Llama-3-8B-Instruct
DATA=selfinst-llama3-4res
DATA_DIR=data/llama3

python3 eval_aspects.py \
    --checkpoint $CKPT \
    --with-aspects True \
    --input-file ./$DATA_DIR/$DATA-$i.jsonl \
    --output-file ./$DATA_DIR/$DATA-reward-BSM-$i.jsonl \
```

If you have access to GPT-4, you can also use it to derive the supervision.
```shell
DATA=selfinst-llama3-4res
DATA_DIR=data/llama3

python3 eval_aspects_online.py \
    --process-num 20 \
    --input-file ./$DATA_DIR/$DATA.jsonl \
    --output-file ./$DATA_DIR/$DATA-reward-online.jsonl \
```

## Self-Correct

Run the following script to derive self-correction supervision.
```shell
i=$1
export CUDA_VISIBLE_DEVICES=$((i))
CKPT=/path/to/Meta-Llama-3-8B-Instruct
DATA=selfinst-llama3-4res-wtaspects
DATA_DIR=data/llama3

python3 reflect_response.py \
    --checkpoint $CKPT \
    --input-file ./$DATA_DIR/$DATA.jsonl \
    --output-file ./$DATA_DIR/$DATA-reflection.json \
```


## Token-level DPO

We have adapted [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) in `LLaMA-Factory-tdpo` which support token-level DPO.

Please format your preference data in alpaca style, refering to the example of ./data/tdpo_demo.json

Please refer to the yaml file ./examples/train_lora/llama3_lora_tdpo.yaml to format your setting file.

For token-level dpo training, you can run the following script:
```bash
llamafactory-cli train examples/train_lora/llama3_lora_tdpo.yaml
```