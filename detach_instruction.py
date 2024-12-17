import os
import re
import time
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--random-seed",
    type=int,
    default=42,
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/data/oss_bucket_0/huanghui/Meta-Llama-3-8B-Instruct",
)
parser.add_argument(
    "--demo-file",
    type=str,
    default="./infobench-data/InfoBench.jsonl",
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
    "--drop-ratio",
    type=float,
    default=0.3
)
parser.add_argument(
    "--response-num",
    type=int,
    default=1,
)
parser.add_argument(
    "--start-idx",
    type=int,
    default=0,
)
parser.add_argument(
    "--total-num",
    type=int,
    default=2000,
)
parser.add_argument(
    "--generate-negative",
    type=str,
    choices=("True", "False"),
    default="True",
)
args = parser.parse_args()

def detach_instruction(demo_lines_i, demo_lines_o, lines, model, tokenizer):

    def format_prompt_detachment(demo_lines, line, tokenizer):

        final_instruction = "Please break down the instruction into multiple constraints and an input, following the provided examples. Do not generate any other openings, closings or other uncessary explanations.\n"
        random.shuffle(demo_lines)
        for i, demo_line in enumerate(demo_lines):
            final_instruction += f"Example {i+1}:\n\n"
            if demo_line["input"] != "":
                demo_instruction = demo_line["instruction"] + "\n" + demo_line["input"]
                demo_input = demo_line["input"]
            else:
                demo_instruction = demo_line["instruction"]
                demo_input = None

            final_instruction += f"**Instruction**:\n{demo_instruction}\n"
            final_instruction += "**Constraints**:\n"
            for j, aspect in enumerate(demo_line["decomposed_questions"]):
                final_instruction = final_instruction + "##" + str(j+1) + ". "+ aspect + "\n"
            if demo_input is not None:
                final_instruction += f"**Input**:\n{demo_input}\n\n"
            else:
                final_instruction += f"**Input**:\nNone\n\n"
        
        final_instruction += "Below is the instruction that needs to be broken down:\n"
        instruction = line["instruction"]
        final_instruction += f"**Instruction**:\n{instruction}\n\n"

        final_instruction = tokenizer.apply_chat_template([{"role": "user", "content": final_instruction}], tokenize=False, add_generation_prompt=True)

        final_instruction += "**Constraints**:\n"

        return final_instruction

    def post_process_detachment(text):

        aspects = []
        try:
            if "**Input**:" in text:
                input = text.split("**Input**:")[1].strip()
                if input.startswith("None"):
                    input = None
                text = text.split("**Input**:")[0]
            else:
                input = None
            aspects = re.split(r"##[0-9]+.", text)[1:]
            aspects = [aspect.strip() for aspect in aspects]
            assert len(aspects) >= 2 # 至少应该有2个aspect            
        except:
            # print(f"Post-processing detachement result failed! Input is {text}")
            aspects = []
            input = None

        return {"aspects": aspects, "input": input}

    prompts = []
    
    for line in lines:
        # 排序靠前的demo_lines会把短的instruction映射为多个aspects
        demo_lines = random.sample(demo_lines_i[:5], k=3) + random.sample(demo_lines_o[:20], k=2)
        prompt = format_prompt_detachment(demo_lines, line, tokenizer)
        prompts.append(prompt)

    sampling_params = SamplingParams(max_tokens=4096, temperature=0.1, seed=args.random_seed)
    print("Now Start to Detach Instructions.")
    outputs = model.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]

    outputs = [post_process_detachment(output) for output in outputs]

    for i, output in enumerate(outputs):
        lines[i]["aspects"] = output["aspects"]
        lines[i]["input"] = output["input"]
    
    new_lines = []
    for line in lines:
        # 如果aspects小于2，说明post_process的时候出现了问题
        if len(line['aspects']) >=2:
            new_lines.append(line)

    return new_lines

def noise_aspect(aspect, drop_ratio=0.5, sub_ratio=0.25, neg_ratio=0.25):

    def substitute_aspect(aspect):

        def parse_response(response):
            if re.match(r"\*\*New Constraint\*\*:", response):
                response = response.split("**New Constraint**:")[1].strip()
            else:
                raise Exception(f"Parsing failed! Response is {response}")
            return response

        prompt = f"""The following is a question used to apply constraint for generating an instruction. Please generate a new constraint question which specifies a constraint deviates from the original one. 
Please generate the constraint question with the following format: '**New Constraint**: [new constraint here]. Do not generate any other openings, closings or explanations.
**Original Constraint**: {aspect}"""

        messages = [{"role": "user", "content": prompt}]
        response = chat_completion_openai(model="/data/oss_bucket_0/huanghui/Meta-Llama-3-8B-Instruct", messages=messages, max_tokens=4096, temperature=0.5)

        new_aspect = parse_response(response)

        return new_aspect

    def negate_aspect(aspect):

        def parse_response(response):
            if re.match(r"\*\*New Constraint\*\*:", response):
                response = response.split("**New Constraint**:")[1].strip()
            else:
                raise Exception(f"Parsing failed! Response is {response}")
            return response

        prompt = f"""The following is a question used to apply constraint for generating an instruction. Please generate a new constraint question which specifies a constraint on the contrary. 
Please generate the constraint question with the following format: '**New Constraint**: [new constraint here]. Do not generate any other openings, closings or explanations.
**Original Constraint**: {aspect}"""

        messages = [{"role": "user", "content": prompt}]
        response = chat_completion_openai(model="/data/oss_bucket_0/huanghui/Meta-Llama-3-8B-Instruct", messages=messages, max_tokens=4096, temperature=0.5)

        new_aspect = parse_response(response)

        return new_aspect
    
    assert drop_ratio + sub_ratio + neg_ratio == 1.0

    random_num = random.random()
    if random_num < drop_ratio:
        return ""
    elif drop_ratio <= random_num <= drop_ratio + sub_ratio:
        return substitute_aspect(aspect)
    else:
        return negate_aspect(aspect)


def create_noised_aspects(line, drop_ratio=0.3):

    noised_aspects = []

    if len(line["aspects"]) < 2:
        line["noised_aspects"] = []
        return line      

    drop_num = round(len(line["aspects"]) * drop_ratio)
    if drop_num == 0:
        drop_num = 1 # at least drop one aspect

    dropped_indices = set(random.sample(np.arange(len(line["aspects"]))[1:].tolist(), drop_num))

    for j, aspect in enumerate(line["aspects"]):
        
        # 对第0个aspect不加噪，否则生成的回复偏离指令太远
        if j not in dropped_indices:
            noised_aspects.append(aspect)
        else:
            noised_aspect = noise_aspect(aspect)
            if noised_aspect != "":
                noised_aspects.append(aspect)
            
    line["noised_aspects"] = noised_aspects

    return line

def create_instructions(demo_lines_i, demo_lines_o, lines, model, tokenizer):

    def format_prompt_instruction(demo_lines, line, tokenizer, aspect_key="aspects"):

        final_instruction = "Please combine the provided input and constraints into an instruction, following the provided examples. \n"
        random.shuffle(demo_lines)
        for i, demo_line in enumerate(demo_lines):
            final_instruction += f"\n##Example {i+1}##:\n"
            if demo_line["input"] != "":
                demo_input = demo_line["input"]
                final_instruction += f"\n**Input**:\n{demo_input}\n"
                demo_instruction = demo_line["instruction"] + "\n" + demo_line["input"]
            else:
                demo_instruction = demo_line["instruction"]
                final_instruction += f"\n**Input**:\nNone\n"
            final_instruction += "\n**Constraints**:\n"
            for j, aspect in enumerate(demo_line["decomposed_questions"]):
                final_instruction = final_instruction + "##" + str(j+1) + ". "+ aspect + "\n"
            final_instruction += f"\n**Instruction**:\n{demo_instruction}\n\n" 

        final_instruction += "Below are the input and constraints for constructing your Instruction:\n"
        if line['input'] is not None:
            input = line['input']
            final_instruction += f"\n**Input**:\n{input}\n"
        else:
            final_instruction += f"\n**Input**:\nNone\n"
        final_instruction += "\n**Constraints**:\n"
        for j, aspect in enumerate(line[aspect_key]):
            final_instruction = final_instruction + "##" + str(j+1) + ". "+ aspect + "\n"

        final_instruction += "Please generate the instruction directly. Please incorporate the content of the input inside the instruction if it is not None. Do not generate any other openings or closings."
        final_instruction = tokenizer.apply_chat_template([{"role": "user", "content": final_instruction}], tokenize=False, add_generation_prompt=True)

        final_instruction += f"**Instruction**:\n"
        return final_instruction

    def post_process_instruction(text):
        if re.search(r"\*\*[^\s]+\*\*", text):
            # print("An exceptional instruction with **:\n"+text)
            return ""
        else:
            return text

    prompts = []
    noised_prompts = []
    
    for line in lines:
        # 排序靠后的demo_lines会把少数aspects组合为一条很长的指令
        if line["input"] != None:
            demo_lines = random.sample(demo_lines_i[-10:], k=2)
        else:
            demo_lines = random.sample(demo_lines_o[-40:], k=2)

        if args.generate_negative == "True":
            line = create_noised_aspects(line, drop_ratio=args.drop_ratio)

        prompt = format_prompt_instruction(demo_lines, line, tokenizer)
        prompts.append(prompt)

        if args.generate_negative == "True":
            noised_prompt = format_prompt_instruction(demo_lines, line, tokenizer, aspect_key="noised_aspects")
            noised_prompts.append(noised_prompt)

    sampling_params = SamplingParams(max_tokens=4096, temperature=0.1, seed=args.random_seed)
    print("Now Start to Create Instructions.")
    outputs = model.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]

    if args.generate_negative == "True":
        print("Now Start to Create Noised Instructions.")
        noised_outputs = model.generate(noised_prompts, sampling_params)
        noised_outputs = [output.outputs[0].text for output in noised_outputs]

    outputs = [post_process_instruction(output) for output in outputs]
    if args.generate_negative == "True":
        noised_outputs = [post_process_instruction(output) for output in noised_outputs]

    for i in range(len(lines)):
        ori_instruction = lines[i]['instruction']
        del(lines[i]["instruction"])
        lines[i]['ori_instruction'] = ori_instruction
        lines[i]['instruction'] = outputs[i]
        if args.generate_negative == "True":
            lines[i]["noised_instruction"] = noised_outputs[i]

    new_lines = []
    for line in lines:
        # 如果instruction为空，那么说明post_process出了问题
        if line['instruction'] != "" and (args.generate_negative=="False" or line["noised_instruction"] != ""):
            new_lines.append(line)

    return new_lines

def create_responses(lines, model, tokenizer, response_num):

    instructions = [line["instruction"] for line in lines]
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in instructions]

    if args.generate_negative == "True":
        noised_instructions = [line["noised_instruction"] for line in lines]
        noised_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in noised_instructions]
    
    sampling_params = SamplingParams(max_tokens=4096, temperature=0.5, n=response_num)

    print("Now Start to Create Responses.")
    outputs = model.generate(prompts, sampling_params)
    if args.generate_negative == "True":
        print("Now Start to Create Noised Responses.")
        noised_outputs = model.generate(noised_prompts, sampling_params)

    responses = [[o.text.strip() for o in output.outputs] for output in outputs]
    if args.generate_negative == "True":
        noised_responses = [[o.text.strip() for o in output.outputs] for output in noised_outputs]

    new_lines = []
    for i, line in enumerate(lines):
        line['responses'] = responses[i]

        if args.generate_negative == "True":
            line['noised_responses'] = noised_responses[i]

        # 如果instruction为空，或者aspects过少，那么说明post_process出了问题
        if line['instruction'] != "" and len(line['aspects']) >= 2 and (args.generate_negative=="False" or line["noised_instruction"] != ""):
            new_lines.append(line)

    return new_lines

def load_demo_file(demo_file):

    demo_file_i = demo_file.rstrip(".jsonl") + "-input.json"
    demo_file_o = demo_file.rstrip(".jsonl") + "-noinput.json"

    fdemo_i = open(demo_file_i, "r")
    demo_lines_i = json.load(fdemo_i)

    fdemo_o = open(demo_file_o, "r")
    demo_lines_o = json.load(fdemo_o)
    
    return demo_lines_i, demo_lines_o

if __name__ == "__main__":
    
    demo_lines_i, demo_lines_o = load_demo_file(args.demo_file)

    fin = open(args.input_file, "r")
    input_lines = json.load(fin)[args.start_idx:]
    
    lines = [{'instruction': line['conversations'][0]['value']} for line in input_lines][args.total_num*args.random_seed: args.total_num*(args.random_seed+1)]

    model = LLM(model=args.checkpoint, tensor_parallel_size=1, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)

    random.seed(args.random_seed) # vllm初始化会重置随机数种子

    lines = detach_instruction(demo_lines_i, demo_lines_o, lines, model, tokenizer)
    lines = create_instructions(demo_lines_i, demo_lines_o, lines, model, tokenizer)
    lines = create_responses(lines, model, tokenizer, args.response_num)

    sample_lines = random.sample(lines, k=3)
    for i in range(len(sample_lines)):
        sample_line = sample_lines[i]
        print(f"*************Sampled Line {i}*************")
        print(json.dumps(sample_line, indent=4))
    print("Totally "+str(len(lines))+" lines after processing.")

    with open(args.output_file, "w") as fout:
        for line in lines:
            fout.write(json.dumps(line) + "\n")