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
from train_all import get_args

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

#     # 无解释信息的情况
#     full_prompt = f"""You are a advanced model trained for vulnerability detection. Your task is to examine real open-source code functions to determine if there are any vulnerabilities present. Focus on the most critical parts of the functions that are most relevant to potential vulnerabilities.
#
# ### Question:
# Does the following function contain any vulnerabilities?
#
# ### Function Code:
# {code}
#
# ### Response:
# """
    return full_prompt


if __name__ == "__main__":
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
    # peft_model_output_dir = "./checkpoints_11u20/RQ1_3_cl_instruct/checkpoint-2668"
    model = PeftModel.from_pretrained(model, peft_model_output_dir)

    dataset = load_dataset("json", data_files=args.test_data_path, split="train")
    i = 0
    for data_point in tqdm(dataset):
        code = data_point["func"]
        label = data_point["target"]
        message = data_point["message"]
        explanation = data_point["Explanation"]
        detail = re.search(r'\[detail\]([\s\S]*)', explanation).group(1)

        target = "vulnerable" if int(label) else "non-vulnerable"

        reference_text = f"""[label]
This function is {target}.

[detail]{detail}
"""
#         reference_text = f"""[label]
# This function is {target}.
# """
        test_prompt = get_prompt(code, message=None)

        if i == 0:
            print(test_prompt)

        result = inference(test_prompt)
        reference = reference_text
        result_list.append(result)
        reference_list.append(reference)
        output_list.append([code, result, reference])
        if i == 0:
            print("[Generated]")
            print(result)
            print("[Reference]")
            print(reference)
            print("[NEXT]")
            i = i + 1

    list_to_csv_with_headers(output_list, args.result_path)
