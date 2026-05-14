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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../CodeLlama-13b-Instruct-hf")
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


def list_to_csv_with_headers(list, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['code', 'generated', 'reference'])
        csv_writer.writerows(list)


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
    os.makedirs(args.result_path.rsplit("/", 1)[0], exist_ok=True)
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

    peft_model_output_dir = args.output_dir + "/peft"
    model = PeftModel.from_pretrained(model, peft_model_output_dir)

    code = args.code
    if args.code is None:
        with open(args.code_path, 'r') as f:
            code = f.read()
    test_prompt = get_prompt(code, message=None)
    result = inference(test_prompt)
    print(result)
