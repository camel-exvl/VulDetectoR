import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import tree_sitter_rust
from openai import OpenAI, BadRequestError
from tqdm import tqdm
from tree_sitter import Language, Parser

api_keys = os.getenv("KEY").split(",")
model = os.getenv("MODEL")
url = os.getenv("URL")
print(f"model: {model}, url: {url}")

parser = Parser()
parser.set_language(Language(tree_sitter_rust.language(), "rust"))
osv_df = pd.read_json("dataset_merged.jsonl", lines=True)
cwe_df = pd.read_csv("cwe.csv", encoding="utf-8", dtype=str, index_col=False)
basic_prompt = """You are a seasoned software vulnerability security expert with rich experience in vulnerability detection and analysis. Your task is to locate the statements in the given real-world open source code functions that are most closely related to the vulnerability, taking into account the relevant vulnerability description and vulnerability label information.
Then, provide a concise explanation related to the vulnerability. To ensure easy understanding, keep the localization and explanation brief.
The function code will be given under the [code] tag, the label of the function will be given under the [label] tag, and the description of the vulnerability will be given under [description] tag.
Please output the original statements of the code which are related to the vulnerability under [statement] tag (output ONLY the code, do not output the line number), and output the explanation of details under [detail] tag. When outputting the results, please strictly follow the templates defined in the example below. Please concentrate solely on vulnerabilities that arise directly from the code itself based on its implementation.

### input

[code]
fn from(other: Py<T>) -> Self {
        let Py(ptr, _) = other;
        Py(ptr, PhantomData)
    }

[label]
This function is vulnerable.

[description]
Reference counting error in pyo3
An issue was discovered in the pyo3 crate before 0.12.4 for Rust. There is a reference-counting error and use-after-free in From<Py<T>>.

### output

[statement]
let Py(ptr, _) = other;

[detail]
The vulnerability is related to a potential use-after-free issue. The function takes ownership of 'other' and destructures it to extract 'ptr'. However, if 'other' is dropped or freed after this operation, 'ptr' could still be used, leading to a use-after-free scenario. This is because 'ptr' might reference memory that has already been freed, and any subsequent operations using 'ptr' could result in undefined behavior.

### input

[code]
fn py(&self) -> Python {
        unsafe { Python::assume_gil_acquired() }
    }

[label]
This function is non-vulnerable.

[description]
Reference counting error in pyo3
An issue was discovered in the pyo3 crate before 0.12.4 for Rust. There is a reference-counting error and use-after-free in From<Py<T>>.

### output

[statement]

[detail]
This function is non-vulnerable. The function safely assumes the Global Interpreter Lock (GIL) is acquired in a controlled manner, and there is no evidence of memory being referenced after it has been freed. The use of unsafe is justified and does not lead to any memory corruption or use-after-free scenarios.

### input"""


def request_gpt(prompt: str):
    retry_time = 2
    while True:
        try:
            client = OpenAI(api_key=random.choice(api_keys), base_url=url)
            completion = (client.chat.completions.create(
                model=model,
                stream=False,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
                top_p=1,
            ))
            return completion
        except Exception as e:
            tqdm.write(f"Error: {e}")
            if isinstance(e, BadRequestError):
                if e.status_code == 400 and ("length" in e.message or "max_seq_len" in e.message):
                    return None
            time.sleep(retry_time)
            retry_time *= 2
            if retry_time > 64:
                raise e


def explanation(row: dict):
    osv_row = osv_df[osv_df["commit_id"] == row["commit_id"]]
    assert osv_row.shape[0] == 1
    osv_row = osv_row.iloc[0].to_dict()
    row["message"] = osv_row["commit_message"]
    vulnerable = f"This function is {'vulnerable' if row['target'] == 1 else 'non-vulnerable'}.\n"
    prompt = f"[code]\n{row['func'].strip()}\n\n[label]\n{vulnerable}\n[description]\n{osv_row['summary'].strip()}\n{osv_row['details'].strip()}\n\n### output\n"
    prompt = f"{basic_prompt}\n\n{prompt}"
    completion = request_gpt(prompt)
    if completion is None:
        tqdm.write(f"#{row['idx']} Error: Too long")
        row["Explanation"] = "<**Too long**>"
        return row, 0
    result = completion.choices[0].message.content
    if row["idx"] == 0:
        tqdm.write(f"{prompt}\n\n{result}")
    row["Explanation"] = result
    return row, completion.usage.total_tokens


def run(filename: str, output_file: str) -> None:
    log_file = str(output_file).split(".")[0] + "_log.json"
    start = 0
    total_tokens = 0
    if os.path.exists(log_file):
        with open(log_file, "r") as log:
            log_data = json.load(log)
            start = log_data["start"]
            total_tokens = log_data["total_tokens"]
            tqdm.write(f"Start from {start}")

    input_df = pd.read_json(filename, lines=True)

    with tqdm(total=input_df.shape[0], desc="Extracting", unit="row") as pbar:
        pbar.update(start)
        batch_size = 32

        for batch_start in range(start, len(input_df), batch_size):
            batch_end = min(batch_start + batch_size, len(input_df))
            current_batch = input_df.iloc[batch_start:batch_end]

            tokens = 0
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = []
                for _, row in current_batch.iterrows():
                    # if row["hash"] != "68045eb99c6bdbbe8e448f9159ac324b":  # example
                    #     futures.append(None)
                    #     continue
                    futures.append(executor.submit(explanation, row.to_dict()))
                # 按顺序收集结果
                results = []
                for future in futures:
                    if future is None:
                        results.append(None)
                    else:
                        result, token = future.result()
                        tokens += token
                        results.append(result)

            # 按原顺序写入结果
            total_tokens += tokens
            with open(output_file, "a+", newline="", encoding="utf-8") as jsonfile:
                for i_in_batch, (idx, row) in enumerate(current_batch.iterrows()):
                    if results[i_in_batch] is None:
                        assert False, "Impossible"
                    json.dump(results[i_in_batch], jsonfile, ensure_ascii=False)
                    jsonfile.write("\n")
            with open(log_file, "w") as log:
                json.dump({"start": batch_end, "total_tokens": total_tokens}, log, indent=4)
            # if tokens > 0:
            #     tqdm.write(
            #         f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Batch {batch_start}-{batch_end} finished, total tokens: {tokens}")
            pbar.update(batch_end - batch_start)


def check(filename: str, output_file: str):
    input_df = pd.read_json(filename, lines=True)
    output_df = pd.read_json(output_file, lines=True)
    print(f"Input: {input_df.shape[0]}, Output: {output_df.shape[0]}")
    for i, (idx, row) in enumerate(input_df.iterrows()):
        assert row["idx"] == output_df.iloc[i]["idx"]
    print("Check passed")


if __name__ == "__main__":
    run("ds_11110_relabeled_11.jsonl", "explanation_11.jsonl")
    check("ds_11110_relabeled_11.jsonl", "explanation_11.jsonl")
