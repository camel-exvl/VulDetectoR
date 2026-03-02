import difflib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import pandas as pd
import tree_sitter_rust
from openai import BadRequestError, OpenAI
from tqdm import tqdm
from tree_sitter import Language, Node, Parser

api_keys = os.getenv("KEY").split(",")
model = sys.argv[1]
# config = sys.argv[2]
config = "11110"
config = tuple([bool(int(i)) for i in config])  # description, message, code, patch, cwe
url = os.getenv("URL")
model_name = model.split("/")[0].lower()
print(f"model: {model}, url: {url}, config: {config}")

parser = Parser()
parser.set_language(Language(tree_sitter_rust.language(), "rust"))
osv_df = pd.read_json("dataset_merged.jsonl", lines=True)
if config[4]:
    cwe_df = pd.read_csv("cwe.csv", encoding="utf-8", dtype=str, index_col=False)


def generate_prompt():
    relevant = ""
    relevant_text = ["vulnerability description", "commit message", "function code", "patch information",
                     "CWE type information"]
    tag = ""
    tag_text = ["the description of the vulnerability will be given under [description] tag",
                "the commit message will be given under the [message] tag",
                "the original function code will be given under the [code] tag",
                "the patch of the function will be given under [patch] tag",
                "the CWE Description will be given under the [cwe] tag"]

    selected_index = [i for i, j in enumerate(config) if j]
    if len(selected_index) == 1:
        relevant = relevant_text[selected_index[0]]
        tag = tag_text[selected_index[0]]
    elif len(selected_index) == 2:
        relevant = f"{relevant_text[selected_index[0]]} and {relevant_text[selected_index[1]]}"
        tag = f"{tag_text[selected_index[0]]} and {tag_text[selected_index[1]]}"
    elif len(selected_index) >= 3:
        relevant = ", ".join(
            relevant_text[i] for i in selected_index[:-1]) + f", and {relevant_text[selected_index[-1]]}"
        tag = ", ".join(tag_text[i] for i in selected_index[:-1]) + f", and {tag_text[selected_index[-1]]}"
    else:
        raise ValueError("Invalid config")
    tag = tag[0].upper() + tag[1:]
    prompt = f"You are a seasoned software vulnerability security expert with rich experience in vulnerability detection and analysis. Your task is to check whether the modifications in the given real-world open source code functions is related to the vulnerability, taking into account the relevant {relevant}.\nThen, provide a concise explanation related to the vulnerability. To ensure easy understanding, keep the explanation brief.\n{tag}.\n"
    prompt += """Please output the label(vulnerable or non-vulnerable) of the function under [label] tag, and output the explanation of details under [detail] tag. When outputting the results, please strictly follow the templates defined in the example below. Please concentrate solely on vulnerabilities that arise directly from the code itself based on its implementation, ignoring unrelated issues such as library changes and external usage context. Only consider changes that directly modify, fix, or affect the original function provided under [code], and ignore any new functions, helper functions, unrelated added code, or code outside the original function scope, even if they appear in the patch. Do not let unrelated additions influence your judgment.

### input

"""
    if config[0]:
        prompt += """[description]
Reference counting error in pyo3
An issue was discovered in the pyo3 crate before 0.12.4 for Rust. There is a reference-counting error and use-after-free in From<Py<T>>.

"""
    if config[1]:
        prompt += """[message]
Merge pull request #1297 from davidhewitt/pyobject-from-py

py: fix reference count bug in From(Py<T>) for PyObject

"""
    if config[2]:
        prompt += """[code]
fn from(other: Py<T>) -> Self {
        let Py(ptr, _) = other;
        Py(ptr, PhantomData)
    }

"""
    if config[3]:
        prompt += """[patch]
@@ -498,2 +499 @@
-        let Py(ptr, _) = other;
-        Py(ptr, PhantomData)
+        unsafe { Self::from_non_null(other.into_non_null()) }

"""
    if config[4]:
        prompt += """[cwe]
It is related to ['CWE-416']. CWE-416 Use After Free: The product reuses or references memory after it has been freed. At some point afterward, the memory may be allocated again and saved in another pointer, while the original pointer references a location somewhere within the new allocation. Any operations using the original pointer are no longer valid because the memory belongs to the code that operates on the new pointer. 

"""
    prompt += """### output

[label]
This function is vulnerable.

[detail]
The vulnerability is related to a potential use-after-free issue. The function takes ownership of 'other' and destructures it to extract 'ptr'. However, if 'other' is dropped or freed after this operation, 'ptr' could still be used, leading to a use-after-free scenario. This is because 'ptr' might reference memory that has already been freed, and any subsequent operations using 'ptr' could result in undefined behavior.

### input"""
    return prompt


basic_prompt = generate_prompt()


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
            if retry_time > 128:
                raise e


def find_function(code: str, func: str) -> Union[Node, None]:
    code = re.sub(r"(?<!r#)\btry!", r"r#try!", code)
    tree = parser.parse(code.encode("utf-8"))

    def dfs(cur: Node) -> Union[Node, None]:
        if cur.type == "mod_item":  # ignore test module
            identifier = cur.children[-2]
            assert identifier.type == "identifier"
            if b"test" in identifier.text:
                return None
        for node in cur.children:
            if node.type == "function_item":
                # ignore #[test] functions
                if node.prev_sibling is not None and node.prev_sibling.type == "attribute_item" and b"test" in node.prev_sibling.text:
                    continue
                if node.text is not None and node.text.decode("utf-8") == func:
                    return node
            elif node.children:
                if (ret := dfs(node)) is not None:
                    return ret
        return None

    return dfs(tree.root_node)


def check_func_llm(row: dict):
    row["llm"] = -1

    osv_row = osv_df[osv_df["commit_id"] == row["commit_id"]]
    assert osv_row.shape[0] == 1
    osv_row = osv_row.iloc[0].to_dict()
    file_changed = None
    for i in osv_row["files_changed"]:
        if i["file_path"] == row["file_path"]:
            file_changed = i
            break
    assert file_changed is not None
    func = find_function(file_changed["code_before"], row["func"])
    assert func is not None

    # get patch
    diff = list(difflib.unified_diff(file_changed["code_before"].splitlines(), file_changed["code_after"].splitlines(),
                                     lineterm="", n=0))
    patch_info = ""
    add_flag = False
    for line in diff:
        if line.startswith("@@"):
            # Attention: diff信息中的行号是从1开始的，而tree-sitter中的行号是从0开始的
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            line_start_before = int(match.group(1)) - 1
            line_end_before = line_start_before + int(match.group(2) or 1) - 1
            line_start_after = int(match.group(3)) - 1
            line_end_after = line_start_after + int(match.group(4) or 1) - 1
            if (
                    func.start_point[0] <= line_start_before <= func.end_point[0]
                    or func.start_point[0] <= line_end_before <= func.end_point[0]
                    or line_start_before <= func.start_point[0] and line_end_before >= func.end_point[0]):
                add_flag = True
            else:
                add_flag = False
        if add_flag:
            patch_info += line + "\n"

    prompt = f"{basic_prompt}\n\n"
    if config[0]:
        prompt += f"[description]\n{osv_row['summary'].strip()}\n{osv_row['details'].strip()}\n\n"
    if config[1]:
        prompt += f"[message]\n{osv_row['commit_message'].strip()}\n\n"
    if config[2]:
        prompt += f"[code]\n{row['func'].strip()}\n\n"
    if config[3]:
        prompt += f"[patch]\n{patch_info.strip()}\n\n"
    if config[4]:
        assert len(row["cwe_id"]) > 0
        cwe_info = ""
        for i, cwe in enumerate(row["cwe_id"]):
            cwe_id = cwe.split("-")[1]
            name = cwe_df[cwe_df["CWE-ID"] == cwe_id]["Name"].values[0]
            description = cwe_df[cwe_df["CWE-ID"] == cwe_id]["Description"].values[0]
            cwe_info += f"{cwe} {name}: {description} "
        prompt += f"[cwe]\nIt is related to {row['cwe_id']}. {cwe_info.strip()}\n\n"
    prompt += "### output\n"

    if row["idx"] == 0 or row["idx"] == 9 and config[4]:
        tqdm.write(prompt)

    # 使用LLM标注
    completion = request_gpt(prompt)
    if completion is None:
        tqdm.write(f"#{row['idx']} Error: Too long")
        return row, 0
    result = completion.choices[0].message.content
    if row["idx"] == 0 or row["idx"] == 9 and config[4]:
        tqdm.write(result)
    try:
        label = re.search(r"\[label\]\s*(.*?)\s*(?=\[detail\]|\Z)", result, re.DOTALL).group(1)
        detail = re.search(r"\[detail\]\s*(.*)", result, re.DOTALL).group(1)
        vulnerable_re = re.search(r"(?<!non-|not\s)vulnerable\b", label.lower())
        non_vulnerable_re = re.search(r"\b(non-|not\s)vulnerable\b", label.lower())
        row["detail"] = detail
        valid = 0
        if vulnerable_re:
            row["llm"] = 1
            valid += 1
        if non_vulnerable_re:
            row["llm"] = 0
            valid += 1
        if valid != 1:
            raise ValueError(f"#{row['idx']} Invalid response: {result.encode('utf-8')}, valid: {valid}")
    except Exception as e:
        raise ValueError(f"#{row['idx']} Error: {e}, response: {result.encode('utf-8')}")
    # tqdm.write(f"#{row['idx']} result: {label}, parsed: {row['llm']}")
    return row, completion.usage.total_tokens


def run(filename: str) -> None:
    os.makedirs(model_name, exist_ok=True)
    output_file = f"{model_name}/{str(filename).split('.')[0]}_llm_{''.join(map(str, map(int, config)))}.jsonl"
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
    input_df = input_df[input_df["target"] == 1]

    with tqdm(total=input_df.shape[0], desc="Extracting", unit="row") as pbar:
        pbar.update(start)
        batch_size = 1

        for batch_start in range(start, len(input_df), batch_size):
            batch_end = min(batch_start + batch_size, len(input_df))
            current_batch = input_df.iloc[batch_start:batch_end]

            tokens = 0
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = []
                for _, row in current_batch.iterrows():
                    # if row["hash"] != "ab5e4564a6f2dcc2b60b99562b4e8366aa5dad06c9a719a07c1e53b5fd897d0b":  # example
                    #     futures.append(None)
                    #     continue
                    if config[4] and len(row["cwe_id"]) == 0:
                        futures.append(None)
                        continue
                    futures.append(executor.submit(check_func_llm, row.to_dict()))
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
                        continue
                    json.dump(results[i_in_batch], jsonfile, ensure_ascii=False)
                    jsonfile.write("\n")
            with open(log_file, "w") as log:
                json.dump({"start": batch_end, "total_tokens": total_tokens}, log, indent=4)
            # if tokens > 0:
            #     tqdm.write(
            #         f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Batch {batch_start}-{batch_end} finished, total tokens: {tokens}")
            pbar.update(batch_end - batch_start)


def print_info(name: str, df: pd.DataFrame) -> None:
    print(
        f"{name}:\ntotal: {df.shape[0]}, vul: {df[df['target'] == 1].shape[0]}, non-vul: {df[df['target'] == 0].shape[0]}, ratio: {df[df['target'] == 0].shape[0] / df[df['target'] == 1].shape[0] if df[df['target'] == 1].shape[0] != 0 else 0}")


def check(filename: str, cwe: bool = False) -> None:
    input_df = pd.read_json("dataset_extracted.jsonl", lines=True)
    input_df = input_df[input_df["target"] == 1]
    if cwe:
        input_df = input_df[input_df["cwe_id"].apply(lambda x: len(x) > 0)]
    llm_df = pd.read_json(filename, lines=True)
    assert input_df.shape[0] == llm_df.shape[0]
    for i, (idx, row) in enumerate(input_df.iterrows()):
        assert row["idx"] == llm_df.iloc[i]["idx"]
    print("Check passed")


def evaluate(filename: str) -> None:
    manual_df = pd.read_json("dataset_manual_label.jsonl", lines=True)
    llm_df = pd.read_json(filename, lines=True)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i, manual_row in manual_df.iterrows():
        llm_row = llm_df[llm_df["idx"] == manual_row["idx"]]
        assert llm_row.shape[0] == 1
        llm_row = llm_row.iloc[0]
        if manual_row["manual"] == 1:
            if llm_row["llm"] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if llm_row["llm"] == 0:
                TN += 1
            else:
                FP += 1
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    print(f"{filename}: {TP}/{FN} {TN}/{FP}, Precision: {precision}, recall: {recall}, f1: {f1}")


def label(filename: str) -> None:
    output_file = str(filename).split(".")[0] + "_relabeled.jsonl"
    df = pd.read_json("dataset_extracted.jsonl", lines=True)
    print_info("Before", df)

    llm_df = pd.read_json(filename, lines=True)
    zero_idx = llm_df.loc[llm_df["llm"] == 0, "hash"]
    exceed_idx = llm_df.loc[llm_df["llm"] == -1, "hash"]
    df.loc[df["hash"].isin(zero_idx), "target"] = 0
    # df = df[~df["hash"].isin(zero_idx)]
    df = df[~df["hash"].isin(exceed_idx)]

    print(f"Exceed: {exceed_idx.shape[0]}")
    df = df.reset_index(drop=True)
    df["idx"] = df.index
    print_info("After labeling", df)
    df.to_json(output_file, orient="records", lines=True)


# random select osv records for manual label
def select_for_manual_label(num: int = 20) -> None:
    osv_df = pd.read_json("dataset_merged.jsonl", lines=True)
    osv_df = osv_df[osv_df["cwe_id"].apply(lambda x: len(x) > 0)]
    osv_df = osv_df.sample(n=num, random_state=42)

    # get functions of each osv_id
    extracted_df = pd.read_json("dataset_extracted.jsonl", lines=True)
    extracted_df = extracted_df[extracted_df["target"] == 1]
    extracted_df = extracted_df[extracted_df["osv_id"].isin(osv_df["osv_id"])]
    extracted_df["manual"] = 0

    extracted_df.to_json("dataset_manual_label.jsonl", orient="records", lines=True)


# # 根据整个数据集的标注结果生成1:1数据集的重标注结果
# def label_11_based_on_whole() -> None:
#     df = pd.read_json("dataset_extracted_11.jsonl", lines=True)
#     print_info("Before", df)
#     llm_df = pd.read_json("deepseek-ai/dataset_extracted_llm_11110_relabeled.jsonl", lines=True)
#     print_info("LLM", llm_df)
#
#     df = llm_df[llm_df["hash"].isin(df["hash"])]
#     df = df.reset_index(drop=True)
#     df["idx"] = df.index
#
#     print_info("After", df)
#     df.to_json("ds_11110_relabeled_11.jsonl", orient="records", lines=True)
#
# # 根据整个数据集的标注结果生成 LLM标注时只保留漏洞函数而直接删除非漏洞函数 的数据集版本
# def label_d_based_on_whole() -> None:
#     df = pd.read_json("dataset_extracted.jsonl", lines=True)
#     print_info("Before", df)
#     llm_df = pd.read_json("deepseek-ai/dataset_extracted_llm_10110.jsonl", lines=True)
#     print_info("LLM", llm_df)
#
#     zero_idx = llm_df.loc[llm_df["llm"] == 0, "hash"]
#     exceed_idx = llm_df.loc[llm_df["llm"] == -1, "hash"]
#     df = df[~df["hash"].isin(zero_idx)]
#     df = df[~df["hash"].isin(exceed_idx)]
#     df = df.reset_index(drop=True)
#     df["idx"] = df.index
#     print(f"Exceed: {exceed_idx.shape[0]}")
#
#     print_info("After", df)
#     df.to_json("dataset_extracted_llm_10110d_relabeled.jsonl", orient="records", lines=True)

if __name__ == "__main__":
    # run("dataset_extracted.jsonl")
    check("gpt-3.5-turbo/dataset_extracted_llm_11110.jsonl")
    evaluate("gpt-3.5-turbo/dataset_extracted_llm_11110.jsonl")
    # label("deepseek-ai/dataset_extracted_llm_11110.jsonl")

    # for i in ["11110", "01110", "10110", "11100"]:
    #     print(f"config: {i}")
    #     # check(f"deepseek-ai/dataset_extracted_llm_{i}.jsonl")
    #     evaluate(f"deepseek-ai/dataset_extracted_llm_{i}.jsonl")
    #     # label(f"deepseek-ai/dataset_extracted_llm_{i}.jsonl")

    # select_for_manual_label()
