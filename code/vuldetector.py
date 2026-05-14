import os

import sys
from datetime import datetime
import argparse
import re
import string
import csv
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import logging, set_seed
from datasets import load_dataset

BOLD = "\033[1m"
RED = BOLD + "\033[31m"
GREEN = BOLD + "\033[32m"
YELLOW = BOLD + "\033[33m"
BLUE = BOLD + "\033[34m"
MAGENTA = BOLD + "\033[35m"
CYAN = BOLD + "\033[36m"
WHITE = BOLD + "\033[37m"
RESET = "\033[0m"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../CodeLlama-13b-Instruct-hf")
    parser.add_argument("--peft_model_path", type=str, default="./checkpoints/RQ1_3_cl_instruct/peft")
    parser.add_argument("--code", type=str, default=None, help="Rust function code text")
    parser.add_argument("--code_path", type=str, default=None, help="Path of a file containing Rust code")

    parser.add_argument("--load_in_8bit", type=bool, default=False)
    parser.add_argument("--add_eos_token", type=bool, default=True)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--padding_side", type=str, default="left")

    # Tokenizer args
    parser.add_argument("--truncation", type=bool, default=True)
    parser.add_argument("--tokenizer_max_length", type=int, default=2048)
    parser.add_argument("--tokenizer_padding", type=bool, default=False)

    # Training args, default 32/8
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", default=10, type=float)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=2024)

    return parser.parse_args()

args = get_args()

def get_response_content(input_string):
    response_index = input_string.find("### Response:")
    if response_index != -1:
        response_content = input_string[response_index + len("### Response:"):]
        return response_content.strip()
    else:
        return "Something wrong happend in getting the response."


def inference(eval_prompt):
    model_input = tokenizer(eval_prompt, return_tensors="pt", max_length=2048, truncation=True).to("cuda")
    model.eval()
    with torch.no_grad():
        generated_text = tokenizer.decode(model.generate(**model_input, max_new_tokens=1024)[0],
                                          skip_special_tokens=True)

        result = get_response_content(generated_text)

        return result


def get_prompt(code, message):
    if message == None:
        full_prompt = f"""You are a advanced model trained for vulnerability detection. Your task is to examine real open-source code functions to determine if there are any vulnerabilities present, and to indentify the key statements that lead to vulnerabilities, and provide concise and intuitive analysis and explanations. Focus on the most critical parts of the functions that are most relevant to potential vulnerabilities.

### Question:
Does the following function contain any vulnerabilities? Explain your findings clearly, referencing specific parts of the code or key statements that indicate vulnerabilities.

### Function Code:
{code}

### Response:
"""
    else:

        full_prompt = f"""You are an advanced model trained for vulnerability detection. Your task is to examine real open-source code functions and their associated commit message to determine if there are any vulnerabilities present, to identify the key statements that lead to vulnerabilities, and provide concise and intuitive analysis and explanations. Focus on the most critical parts of the functions that are most relevant to potential vulnerabilities.

### Question:
Does the following function contain any vulnerabilities? Explain your findings clearly, referencing specific parts of the code or key statements that indicate vulnerabilities.

### Function Code:
{code}

### Commit Message:
{message}

### Response:
"""
    return full_prompt


if __name__ == "__main__":
    if args.code_path is None and args.code is None:
        print("Please provide either a code file or code text.")
        sys.exit(1)
    set_seed(args.seed)
    logging.set_verbosity_error()
    result_list = []
    reference_list = []
    output_list = []

    base_model = args.model_path
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.peft_model_path)

    code = args.code
    if args.code is None:
        with open(args.code_path, 'r') as f:
            code = f.read()
    test_prompt = get_prompt(code, message=None)
    result = inference(test_prompt)

    label_match = re.search(r'\[label\]([\s\S]*?)\[detail\]', result)
    detail_match = re.search(r'\[detail\]([\s\S]*)', result)
    if not label_match or not detail_match:
        print(result)
    else:
        label_content = label_match.group(1).strip()
        detail_content = detail_match.group(1).strip()

        label_content = label_content.replace("vulnerable", f"{RED}vulnerable{RESET}")
        label_content = label_content.replace(f"non-{RED}vulnerable{RESET}", f"{GREEN}non-vulnerable{RESET}")

        print(f"{YELLOW}[label]{RESET}")
        print(label_content)
        print(f"{YELLOW}[detail]{RESET}")
        print(detail_content)