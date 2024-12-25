# -*- coding: utf-8 -*-
import os
import re
import time
import json
import openai
import random
import argparse
import numpy as np
import timeout_decorator
import multiprocessing
from tqdm import tqdm

parser = argparse.ArgumentParser()
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
    "--api-key",
    type=str,
    default="your-key-here",
)
parser.add_argument(
    "--process-num",
    type=int,
    default=1,
)
args = parser.parse_args()

@timeout_decorator.timeout(1200)    
def OneAPIRequest(user_prompt, model_name, temperature = 0.5):
    client = openai.OpenAI(api_key=args.api_key, base_url="https://idealab.alibaba-inc.com/api/openai/v1")
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
    }
    MAX_RETRY = 5
    res = ""
    for i in range(MAX_RETRY):
        try:
            chat_completion = client.chat.completions.create(model=payload['model'], temperature=temperature, messages=payload['messages'])
            res = chat_completion.choices[0].message.content
            return res
        except timeout_decorator.TimeoutError:
            raise Exception("Stuck! Please re-run this script!")
        except Exception as e:
            if i == MAX_RETRY-1:
                raise Exception("MAX_RETRY exceeded! Please check your codes! ")
            print("Failed! Retrying! The exception is "+str(e))
            time.sleep(5)
            continue
    return res

def evaluate_lines_aspects(line):

    def pre_process_aspects(instruction, response, aspect):
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

        prompt = prompt_template.format(instruction=instruction, response=response, aspect=aspect)
        return prompt

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
    
    line_scores = []
    all_aspect_scores = []
    for response in line["responses"]:
        aspect_scores = []
        for aspect in line['aspects']:
            prompt = pre_process_aspects(line['instruction'], response, aspect)
            response = OneAPIRequest(prompt, model_name="gpt-4o-0513", temperature=0.1)
            score = post_process_aspects(response)
            aspect_scores.append(score)
        all_aspect_scores.append(aspect_scores)
        line_scores.append(round(sum(aspect_scores)/len(aspect_scores), 2))

    line["scores"] = line_scores
    line["aspect_scores"] = all_aspect_scores

    with open(args.output_file, 'a+') as fout:
        fout.write(json.dumps(line) + "\n")
    
    counter.value += 1
    if counter.value % 10 == 0:
        used_time = time.time() - start_time
        remain_time = (len(unfinished_lines) - counter.value) * (used_time / counter.value) / 60
        print("*******************************")
        print(f"{counter.value} lines finished in {used_time:.2f} seconds! {remain_time: .2f} minutes needed before finishing.")
        # print("*******************************")
        # print("sampled line: " + json.dumps(line, indent=4))

def init(c, t):
    global counter
    global start_time
    counter = c
    start_time = t


if __name__ == "__main__":

    manager = multiprocessing.Manager()
    counter = manager.Value("counter", 0)
    start_time = time.time()
    
    fin = open(args.input_file, "r")
    lines = [json.loads(line.strip()) for line in fin.readlines()]

    if os.path.exists(args.output_file):
        fout = open(args.output_file, "r")
        finished_lines = [json.loads(line.strip())["instruction"] for line in fout.readlines()]
        finished_insts = set(finished_lines)

    unfinished_lines = []
    for line in lines:
        if not os.path.exists(args.output_file) or line["instruction"] not in finished_insts:
            unfinished_lines.append(line)
    
    if args.process_num == 1:
        for line in tqdm(unfinished_lines):
            evaluate_lines_aspects(line)
    else:
        pool = multiprocessing.Pool(processes=args.process_num, initializer=init, initargs=(counter, start_time, ))
        pool.map(evaluate_lines_aspects, unfinished_lines)
        pool.close()

    if os.path.exists(args.output_file):
        fout = open(args.output_file, "r")
        finished_lines = [json.loads(line.strip())["instruction"] for line in fout.readlines()]
        finished_insts = set(finished_lines)
