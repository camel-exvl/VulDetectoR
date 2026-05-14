import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from openai import OpenAI, BadRequestError
from tqdm import tqdm

api_keys = os.getenv("KEY").split(",")
model = os.getenv("MODEL")
url = os.getenv("URL")
print(f"model: {model}, url: {url}")

basic_prompt = """### Task
You are a seasoned software vulnerability security expert with rich experience in vulnerability detection and analysis. Your task is to review the provided code involving vulnerabilities and the corresponding vulnerability information, and evaluate them from three independent dimensions: accuracy, clarity, and specificity.

### Instructions
The specific meanings of each indicator are as follows:
Accuracy: Does the explanation correctly identify the main vulnerabilities in the code and its root cause?
Clarity: Does the explanation clearly and understandably describe the vulnerability, its cause, and its potential impact?
Specificity: Does the explanation refer specifically to the function’s behavior or code, regardless of whether those references are correct?

Please review the [explanation] based on the meaning of each indicator and provide scores and reasons for each indicator. For each dimension, assign 1 if the explanation clearly meets the criteria above, and 0 if it does not fully meet the standard or shows noticeable flaws in that area.
The provided reference is not a model explanation. It only states the core issue in the code and is provided solely to help assess the technical correctness of the explanation.
Please strictly follow the templates defined in the example below. 

### Input
[code]
pub fn allow_file<P: AsRef<Path>>(&self, path: P) -> crate::Result<()> {
    let path = path.as_ref();
    push_pattern(&mut self.allowed_patterns.lock().unwrap(), &path)?;
    self.trigger(Event::PathAllowed(path.to_path_buf()));
    Ok(())
  }
[explanation]
The vulnerability arises from the use of wildcard patterns (`*`) in the path, which can lead to unintended directory traversal. On Linux/MacOS, these patterns can bypass the intended scope restrictions, allowing access to neighboring files and subdirectories. On Windows, the impact is limited but still allows access to single-character files in an allowed directory. The issue stems from not properly escaping these special characters in the path.
[reference]
The vulnerability stems from improper handling of special characters in file paths. The function `push_pattern` adds the given path to the allowed patterns without proper escaping of wildcard characters (`*`, `**`, `[a-Z]`). This allows an attacker to bypass file access restrictions by crafting paths with these special characters, potentially accessing neighboring files or subfolders that should be restricted. The issue is platform-dependent, affecting Linux/MacOS more severely than Windows.
### Output
[Accuracy]
1
[Clarity]
0
[Specificity]
0
[Detail]
The explanation accurately identifies the vulnerability as an wildcard pattern misuse issue, which is correct based on the reference. The explanation does not clearly explain how the wildcard patterns actually bypass restrictions. The explanation does not quote any specific function behavior in the given code.

### Input
"""


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


def evaluation(row: dict):
    ret = {"code": row["code"], "generated": row["generated"], "reference": row["reference"], "accuracy": -1,
           "clarity": -1, "specificity": -1}
    detail = re.search(r'\[detail\]([\s\S]*)', row["generated"])
    reference = re.search(r'\[detail\]([\s\S]*)', row["reference"])
    if detail is None or reference is None:
        tqdm.write(f"#{row['idx']} Error: No detail")
        return ret, 0
    detail = detail.group(1).strip()
    reference = reference.group(1).strip()
    prompt = (basic_prompt + "[code]\n" + row[
        "code"] + "\n[explanation]\n" + detail + "\n[reference]\n" + reference + "\n\n### Output")
    completion = request_gpt(prompt)
    if completion is None:
        tqdm.write(f"#{row['idx']} Error: Too long")
        return ret, 0
    result = completion.choices[0].message.content
    try:
        accuracy = re.search(r"\[Accuracy\]\s*([01])", result).group(1)
        clarity = re.search(r"\[Clarity\]\s*([01])", result).group(1)
        specificity = re.search(r"\[Specificity\]\s*([01])", result).group(1)
        detail = re.search(r"\[Detail\]\s*([\s\S]*)", result).group(1)
        assert accuracy in ["0", "1"]
        assert clarity in ["0", "1"]
        assert specificity in ["0", "1"]
        ret["accuracy"] = int(accuracy)
        ret["clarity"] = int(clarity)
        ret["specificity"] = int(specificity)
        ret["detail"] = result
    except Exception as e:
        tqdm.write(f"#{row['idx']} Error: {e}")
        tqdm.write(result)
        return ret, completion.usage.total_tokens
    return ret, completion.usage.total_tokens


# 提取识别正确的正例
def preprocess(filename: str) -> pd.DataFrame:
    input_df = pd.read_csv(filename, usecols=["code", "generated", "reference"])
    df = []
    label_map = {'non-vulnerable': 0, 'vulnerable': 1}
    pattern = re.compile(r'\[label\]\s*\n\s*([^\n.]+)')
    for index, row in input_df.iterrows():
        generated_result = row["generated"].lower()
        reference_result = row["reference"].lower()
        predict_label = pattern.search(generated_result)
        true_label = pattern.search(reference_result)
        if predict_label is not None and true_label is not None:
            try:
                predict_label = label_map[predict_label.group(1).split(" ")[-1]]
                true_label = label_map[true_label.group(1).split(" ")[-1]]
            except:
                print(true_label.group(1))
                continue
            if predict_label == 1 and true_label == 1:
                df.append(row)
    df = pd.DataFrame(df)
    return df


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

    input_df = preprocess(filename)

    with tqdm(total=input_df.shape[0], desc="Extracting", unit="row") as pbar:
        pbar.update(start)
        batch_size = 16

        for batch_start in range(start, len(input_df), batch_size):
            batch_end = min(batch_start + batch_size, len(input_df))
            current_batch = input_df.iloc[batch_start:batch_end]

            tokens = 0
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = []
                for _, row in current_batch.iterrows():
                    # if not row["code"].startswith("pub fn pow(self, exp: u32) -> Self {"):  # example
                    #     futures.append(None)
                    #     continue
                    futures.append(executor.submit(evaluation, row.to_dict()))
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


def eval_all(filename: str):
    df = pd.read_json(filename, lines=True)
    # 计算每个指标的平均值
    accuracy_mean = df["accuracy"].mean()
    clarity_mean = df["clarity"].mean()
    specificity_mean = df["specificity"].mean()
    # 计算每个指标的标准差
    accuracy_std = df["accuracy"].std()
    clarity_std = df["clarity"].std()
    specificity_std = df["specificity"].std()
    print(
        f"Accuracy: {accuracy_mean}, Clarity: {clarity_mean}, Specificity: {specificity_mean}")


def select_for_manual_label(filename: str) -> None:
    df = preprocess(filename)
    # df = df.sample(n=num, random_state=42)
    df["accuracy"] = 0
    df["clarity"] = 0
    df["specificity"] = 0
    df.to_json("evaluation_11_manual.jsonl", orient="records", lines=True)


def eval_manual(filename: str) -> None:
    df = pd.read_json(filename, lines=True)
    auto_df = pd.read_json("evaluation_11_ref.jsonl", lines=True)
    log_file = str(filename).split(".")[0] + "_log.json"
    start = 0
    if os.path.exists(log_file):
        with open(log_file, "r") as log:
            log_data = json.load(log)
            start = log_data["start"]
            tqdm.write(f"Start from {start}")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating", unit="row"):
        if index < start:
            continue
        detail = re.search(r'\[detail\]([\s\S]*)', row["generated"])
        reference = re.search(r'\[detail\]([\s\S]*)', row["reference"])
        if detail is None or reference is None:
            tqdm.write(f"#{row['idx']} Error: No detail")
            continue
        detail = detail.group(1).strip()
        reference = reference.group(1).strip()
        print(f"[code]\n{row['code']}\n[generated]\n{detail}\n[reference]\n{reference}")

        accuracy_ref = auto_df.iloc[index]["accuracy"]
        clarity_ref = auto_df.iloc[index]["clarity"]
        specificity_ref = auto_df.iloc[index]["specificity"]
        assert row["code"] == auto_df.iloc[index]["code"]

        print(f"#{index} AI: {accuracy_ref}, {clarity_ref}, {specificity_ref}. Please input your label:")
        accuracy = input("Accuracy: ")
        clarity = input("Clarity: ")
        specificity = input("Specificity: ")
        assert accuracy in ["0", "1"]
        assert clarity in ["0", "1"]
        assert specificity in ["0", "1"]
        df.at[index, "accuracy"] = int(accuracy)
        df.at[index, "clarity"] = int(clarity)
        df.at[index, "specificity"] = int(specificity)
        print(f"#{index} finished, {accuracy}, {clarity}, {specificity}")
        df.to_json(filename, orient="records", lines=True)
        with open(log_file, "w") as log:
            json.dump({"start": index + 1}, log, indent=4)
    eval_all(filename)


if __name__ == "__main__":
    # run("RQ1_3.csv", "evaluation_11_ref.jsonl")
    # eval_all("evaluation_11_ref.jsonl")

    # select_for_manual_label("RQ1_3.csv")
    # eval_manual("evaluation_11_manual.jsonl")
    # eval_all("evaluation_11_manual.jsonl")

    run("RQ1_3_7B.csv", "evaluation_11_7B.jsonl")
    eval_all("evaluation_11_7B.jsonl")
