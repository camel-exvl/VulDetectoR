import os
import re

from sklearn.metrics import f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # enable this to disable the warning message from huggingface/tokenziers
import sys
from datetime import datetime
import argparse
import torch
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import logging, set_seed
from datasets import load_dataset, Features, Value
from tqdm.auto import tqdm
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="../CodeLlama-7b-Instruct-hf")
    parser.add_argument("--model_path", type=str, default="../CodeLlama-13b-Instruct-hf")
    parser.add_argument("--training_data_path", type=str, default="../dataset/e11dutrain.jsonl")
    parser.add_argument("--val_data_path", type=str, default="../dataset/e11duvalid.jsonl")
    parser.add_argument("--test_data_path", type=str, default="../dataset/e11dutest.jsonl")
    parser.add_argument("--result_path", type=str, default="./result/RQ1_3.csv")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/RQ1_3_cl_instruct")

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
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.add_eos_token = args.add_eos_token
tokenizer.pad_token_id = args.pad_token_id
tokenizer.padding_side = args.padding_side


def run_training(args):
    # Load dataset
    train_dataset_load = load_dataset("json", data_files=args.training_data_path, split="train")
    val_dataset_load = load_dataset("json", data_files=args.val_data_path, split="train")

    train_dataset = train_dataset_load
    eval_dataset = val_dataset_load
    print(len(train_dataset))
    print(len(eval_dataset))

    # Load model
    base_model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=args.load_in_8bit,
        # torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    model.train()

    # LORA configs
    config = LoraConfig(
        # r=16,
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, config)

    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    batch_size = args.batch_size
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = batch_size // per_device_train_batch_size
    output_dir = args.output_dir

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=3e-4,
        fp16=args.fp16,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="epoch",  # if val_set_size > 0 else "no",
        save_strategy="epoch",
        # eval_steps=10000,
        # save_steps=30000,
        output_dir=output_dir,
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True,  # group sequences of roughly the same length together to speed up training
        report_to="none",  # if use_wandb else "none",
        # run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        print("compiling the model")
        model = torch.compile(model)

    trainer.train()
    peft_model_output_dir = args.output_dir + "/peft"
    model.save_pretrained(peft_model_output_dir)
    train_loss = []
    for obj in trainer.state.log_history:
        if 'loss' in obj.keys():
            train_loss.append(obj['loss'])

    with open(args.output_dir + 'loss_logs.txt', 'w') as f:
        for item in train_loss:
            f.write(str(item) + '\n')


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=args.truncation,
        max_length=args.tokenizer_max_length,
        padding=args.tokenizer_padding,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()
    return result


i = 0


def get_prompt(code, message, reference_text):
    if message == None:
        full_prompt = f"""You are a advanced model trained for vulnerability detection. Your task is to examine real open-source code functions to determine if there are any vulnerabilities present, and to indentify the key statements that lead to vulnerabilities, and provide concise and intuitive analysis and explanations. Focus on the most critical parts of the functions that are most relevant to potential vulnerabilities.

### Question:
Does the following function contain any vulnerabilities? Explain your findings clearly, referencing specific parts of the code or key statements that indicate vulnerabilities.

### Function Code:
{code}

### Response:
{reference_text}
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
{reference_text}
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
# {reference_text}
# """
    return full_prompt


def generate_and_tokenize_prompt(data_point):
    code = data_point["func"]
    label = data_point["target"]
    message = data_point["message"]
    explanation = data_point['Explanation']
    detail = re.search(r'\[detail\]([\s\S]*)', explanation).group(1)

    target = "vulnerable" if int(label) else "non-vulnerable"

    reference_text = f"""[label]
This function is {target}.

[detail]{detail}
"""
#     reference_text = f"""[label]
# This function is {target}.
# """

    full_prompt = get_prompt(code, None, reference_text)
    global i
    if (i == 0):
        print(full_prompt)
        i = i + 1

    return tokenize(full_prompt)


if __name__ == "__main__":
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.set_verbosity_info()
    run_training(args)
