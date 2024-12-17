# -*- coding: utf-8 -*-
import os
import re
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/home/hh456524/HFModels/Qwen2-7B-Instruct/",
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
    "--with-aspects",
    type=str,
    choices=("True", "False"),
    default="True",    
)
args = parser.parse_args()

def evaluate_lines_aspects(lines):

    def pre_process_aspects(line):
        prompt_template = """Please act as an impartial and fair judge, analyze the content of the **model response**, and choose "Yes" or "No" to answer whether the subsequent **question** is valid.

The **question** can be understood as a scoring point for the **input instruction** in steps, judging whether a specific part has been met. Therefore, you only need to consider whether the requirement asked by the **question** is established, without paying attention to whether the entire **input instruction** has been fully satisfied.

Please check whether the **model response** has been completed in response to the **question**, fully understand the meaning of the **question**, do not miss small details. Only focus on the current **question**, and do not pay attention to other requirements in the **input instruction**. It must be perfectly and fully completed to be evaluated as "Yes". Even if there is a slight error or ambiguous content, it cannot be "Yes", and there should not be statements such as "basically correct", "mostly correct", "correct under certain conditions", all these situations should be evaluated as "No".

If the text of the **model response** cannot meet the requirements of the current **question** or does not provide information that can be used to answer the **question**, choose "No".

## Output Format
Analysis: xxx
Answer: Yes/No

## Judging Information
**Input Instruction**
{instruction}

**Model Response**
{response}

**Question**
{aspect}

Please analyze and answer whether the **model response** meets the **question**."""

        instruction = line["instruction"]
        responses = line["responses"]
        aspects = line["aspects"]

        prompts = []
        indices = [] # indices 的格式为 [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]，表明了当前的prompt属于第几个response
        assert len(responses) >= 2
        assert len(aspects) >= 2
        for i, response in enumerate(responses):
            for aspect in aspects:
                prompts.append(prompt_template.format(instruction=instruction, response=response, aspect=aspect))
                indices.append(i)
        return prompts, indices

    def post_process_aspects(text):
        if "Answer: Yes" in text or "Answer: **Yes**" in text or "Answer: ** Yes" in text:
            score = 1.0
        elif "Answer: No" in text or "Answer: **No**" in text or "Answer: ** No" in text:
            score = 0.0
        else:
            # print("Score parsing failed! Response is:" + str(text))
            # 大部分的得分都是1.0，因此这里将0.9作为默认得分
            score = 0.9
        return score

    all_prompts = []
    all_indices = []
    for line in lines:
        prompts, indices = pre_process_aspects(line)
        all_prompts.extend(prompts)
        all_indices.extend(indices)

    model = LLM(model=args.checkpoint, tensor_parallel_size=1, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    sampling_params = SamplingParams(max_tokens=4096, n=1, temperature=0.1)

    all_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in all_prompts]
    outputs = model.generate(all_prompts, sampling_params)

    results = [output.outputs[0].text for output in outputs]
    results = [post_process_aspects(result) for result in results]

    FLAG = 0
    scores = []
    all_scores = []
    for i, result in zip(all_indices, results):
        if i != FLAG:
            FLAG = i
            all_scores.append(scores)
            scores = [] # 每个scores对应一个response对于多个aspects的得分
        scores.append(result)

    assert len(scores) != []
    all_scores.append(scores)

    i = 0
    for line in lines:
        aspect_scores = []
        line_scores = []
        for _ in range(len(line["responses"])):
            aspect_scores.append(all_scores[i])
            line_scores.append(round(sum(all_scores[i])/len(all_scores[i]), 2))
            i += 1

        line["scores"] = line_scores
        line["aspect_scores"] = aspect_scores
    
    return lines

def evaluate_lines_noaspects(lines):

    def pre_process_noaspects(line):
        prompt_template = """Review the user’s question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:
- Add 1 point if the response is relevant and provides some information related to the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question, but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective, addressing the user’s question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.
User: {instruction}
<response>{response}</response>
After examining the user’s instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: “Score: <total points>”, without other closings.
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we’ll systematically attribute points based on the outlined criteria."""

        instruction = line["instruction"]
        responses = line["responses"]

        prompts = []
        for response in responses:
            prompts.append(prompt_template.format(instruction=instruction, response=response))

        return prompts

    def post_process_noaspects(text):
        result = re.search(r"[S|s]core: [0-9]+", text)
        try:
            score = float(result.group().split("core:")[1])
            return score
        except:
            # print("Post process faild! Output is "+text)
            return 5.0

    prompts = []
    for line in lines:
        prompts.extend(pre_process_noaspects(line))

    model = LLM(model=args.checkpoint, tensor_parallel_size=1, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    sampling_params = SamplingParams(max_tokens=4096, n=1, temperature=0.1)

    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
    outputs = model.generate(prompts, sampling_params)

    results = [output.outputs[0].text for output in outputs]
    all_scores = [post_process_noaspects(result) for result in results]

    i = 0
    for line in lines:
        line_scores = []
        for _ in range(len(line["responses"])):
            line_scores.append(all_scores[i])
            i += 1

        line["scores"] = line_scores

    return lines


if __name__ == "__main__":
    
    fin = open(args.input_file, "r")
    lines = [json.loads(line.strip()) for line in fin.readlines()]

    if args.with_aspects == "True":
        lines = evaluate_lines_aspects(lines)
    else:
        lines = evaluate_lines_noaspects(lines)

    print("Totally "+str(len(lines))+" lines after processing.")
    sample_lines = random.sample(lines, k=3)
    for i in range(len(sample_lines)):
        sample_line = sample_lines[i]
        print(f"*************Sampled Line {i}*************")
        print(json.dumps(sample_line, indent=4))
    
    with open(args.output_file, "w") as fout:
        for line in lines:
            fout.write(json.dumps(line) + "\n")