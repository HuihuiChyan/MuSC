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
from detach_instruction import args, create_noised_aspects, create_responses, load_demo_file

def post_process_detachment(response):
    aspects = []
    try:
        if "**Input:**" in response:
            input = response.split("**Input:**")[1].strip()
            if input.startswith("None"):
                input = None
            response = response.split("**Input:**")[0]
        else:
            input = None
        aspects = re.split(r"##[0-9]+.", response)[1:]
        aspects = [aspect.strip() for aspect in aspects]
        assert len(aspects) >= 2 # 至少应该有2个aspect            
    except:
        # print(f"Post-processing detachement result failed! Input is {response}")
        aspects = []
        input = None
    return {"aspects": aspects, "input": input}

def create_selfinst(demo_lines_i, demo_lines_o, model, tokenizer):

    def format_prompt_selfinst(demo_lines, tokenizer):

        final_instruction = "Please follow the examples to come up with one task with more than 5 constraints. Please list the constraints first and then provide the task. \nExamples:\n\n"
        random.shuffle(demo_lines)
        for i, demo_line in enumerate(demo_lines):
            if demo_line["input"] != "":
                demo_instruction = demo_line["instruction"] + "\n" + demo_line["input"]
            else:
                demo_instruction = demo_line["instruction"]

            final_instruction += "**Constraints:**\n"
            for j, aspect in enumerate(demo_line["decomposed_questions"]):
                final_instruction = final_instruction + "##" + str(j+1) + ". "+ aspect + "\n"

            final_instruction += f"**Task:**\n{demo_instruction}\n"

        return final_instruction

    def post_process_selfinst(response):
        aspects = []
        try:
            instruction = response.split("**Task:**")[1]
            aspects = response.split("**Task:**")[0]
            aspects = aspects.split("**Constraints:**")[1]
            aspects = re.split(r"##[0-9]+.", aspects)[1:]
            instruction = instruction.strip()
            aspects = [aspect.strip() for aspect in aspects]
        except:
            # print(f"Post-processing failed! Input is {response}")
            aspects = []
            instruction = ""
        return {"instruction": instruction, "aspects": aspects}

    prompts = []
    noised_prompts = []
    
    for i in range(args.total_num):
        demo_lines = random.sample(demo_lines_i, k=2) + random.sample(demo_lines_o, k=2)
        prompt = format_prompt_selfinst(demo_lines, tokenizer)
        prompts.append(prompt)

    print("Now Start to Create Self-Instructions.")
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
    sampling_params = SamplingParams(max_tokens=4096, temperature=0.5)
    outputs = model.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]

    lines = [post_process_selfinst(output) for output in outputs]
    # 为了确保chosen inst和rejected inst使用的demo_line都是一样的，到这里不进行筛选

    return lines

def create_noiseinst(demo_lines_i, demo_lines_o, lines, model, tokenizer):

    def format_prompt_noiseinst(demo_lines, line, tokenizer):

        final_instruction = "Please follow the examples to come up with one task with several constraints. Please list the constraints first and then provide the task. \nExamples:\n\n"
        random.shuffle(demo_lines)
        for i, demo_line in enumerate(demo_lines):
            if demo_line["input"] != "":
                demo_instruction = demo_line["instruction"] + "\n" + demo_line["input"]
            else:
                demo_instruction = demo_line["instruction"]

            final_instruction += "**Constraints:**\n"
            for j, aspect in enumerate(demo_line["decomposed_questions"]):
                final_instruction = final_instruction + "##" + str(j+1) + ". "+ aspect + "\n"

            final_instruction += f"**Task:**\n{demo_instruction}\n"
        
        final_instruction = tokenizer.apply_chat_template([{"role": "user", "content": final_instruction}], tokenize=False, add_generation_prompt=True)

        aspects_prompt = "**Constraints:**\n"
        for j, aspect in enumerate(line["noised_aspects"]):
            aspects_prompt = aspects_prompt + "##" + str(j+1) + ". "+ aspect + "\n"
        final_instruction = final_instruction + aspects_prompt + "\n\n**Task:**\n"

        return final_instruction

    def post_process_noiseinst(response):
        if re.search(r"\*\*[^\s]+\*\*", response):
            # print("An exceptional instruction with **:\n"+response)
            return ""
        else:
            return response

    prompts = []
    noised_prompts = []

    random.seed(args.random_seed) # 为了确保选择的demo_line都是一样的，这里重新初始化一遍种子
    
    for line in lines:
        demo_lines = random.sample(demo_lines_i, k=2) + random.sample(demo_lines_o, k=2)
        line = create_noised_aspects(line)
        prompt = format_prompt_noiseinst(demo_lines, line, tokenizer)
        prompts.append(prompt)

    print("Now Start to Create Noised Instructions.")
    sampling_params = SamplingParams(max_tokens=4096, temperature=0.5)
    outputs = model.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    outputs = [post_process_noiseinst(output) for output in outputs]

    for i in range(len(lines)):
        lines[i]["noised_instruction"] = outputs[i]

    new_lines = []
    for line in lines:
        # 如果instruction为空，或者aspects过少，那么说明post_process出了问题
        if line['instruction'] != "" and len(line['aspects']) >= 2 and (args.generate_negative=="False" or line["noised_instruction"] != ""):
            new_lines.append(line)

    return new_lines

def filt_dup_instructions(lines):
    recipe_cnt = 0
    story_cnt = 0
    travel_cnt = 0
    filted_lines = []
    for line in lines:
        if re.match(r"[A-Z][^\s]+ a [^\.,]*recipe", line["instruction"]):
            recipe_cnt += 1
            if recipe_cnt <= 25:
                filted_lines.append(line)
        elif re.match(r"[A-Z][^\s]+ a [^\.,]*story", line["instruction"]):
            story_cnt += 1
            if story_cnt <= 25:
                filted_lines.append(line)
        elif re.match(r"[A-Z][^\s]+ a [^\.,]*travel", line["instruction"]):
            travel_cnt += 1
            if travel_cnt <= 25:
                filted_lines.append(line)
        else:
            filted_lines.append(line)
    return filted_lines

if __name__ == "__main__":
    
    demo_lines_i, demo_lines_o = load_demo_file(args.demo_file)

    model = LLM(model=args.checkpoint, tensor_parallel_size=1, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)

    random.seed(args.random_seed) # vllm初始化会重置随机数种子

    lines = create_selfinst(demo_lines_i, demo_lines_o, model, tokenizer)
    if args.generate_negative == "True":
        lines = create_noiseinst(demo_lines_i, demo_lines_o, lines, model, tokenizer)
    lines = create_responses(lines, model, tokenizer, args.response_num)
    lines = filt_dup_instructions(lines)

    sample_lines = random.sample(lines, k=3)
    for i in range(len(sample_lines)):
        sample_line = sample_lines[i]
        print(f"*************Sampled Line {i}*************")
        print(json.dumps(sample_line, indent=4))
    print("Totally "+str(len(lines))+" lines after processing.")
    
    with open(args.output_file, "w") as fout:
        for line in lines:
            fout.write(json.dumps(line) + "\n")