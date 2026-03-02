import os.path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

seed = 42


def print_info(name: str, df: pd.DataFrame) -> None:
    print(
        f"{name}:\ntotal: {df.shape[0]}, vul: {df[df['target'] == 1].shape[0]}, non-vul: {df[df['target'] == 0].shape[0]}, ratio: {df[df['target'] == 0].shape[0] / df[df['target'] == 1].shape[0] if df[df['target'] == 1].shape[0] != 0 else 0}")


def divide_df(df: pd.DataFrame, output_prefix: str = "") -> None:
    # # RUS
    # rus = RandomUnderSampler(sampling_strategy=1, random_state=seed)
    # df_data, df_target = rus.fit_resample(df.drop(columns=["idx", "target"]), df["target"])
    # df = pd.concat([df_data, df_target], axis=1)
    # df["idx"] = df.index
    # print_info("After RUS", df)

    # random split
    train, valid = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["target"])
    valid, test = train_test_split(valid, test_size=0.5, random_state=seed, stratify=valid["target"])

    print_info("Train", train)
    print_info("Valid", valid)
    print_info("Test", test)

    # # ROS
    # ros = RandomOverSampler(sampling_strategy=1, random_state=seed)
    # train_data, train_target = ros.fit_resample(train.drop(columns=["idx", "target"]), train["target"])
    # train = pd.concat([train_data, train_target], axis=1)
    # train["idx"] = train.index
    # print_info("Train after ROS", train)

    train.to_json(f"{output_prefix}train.jsonl", orient="records", lines=True)
    valid.to_json(f"{output_prefix}valid.jsonl", orient="records", lines=True)
    test.to_json(f"{output_prefix}test.jsonl", orient="records", lines=True)

    # # 5-fold cross-validation
    # train = pd.concat([train, valid], ignore_index=True)
    # print("\n5-fold:")
    # print_info("Train + Valid", train)
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # for i, (train_index, valid_index) in enumerate(skf.split(train, train["target"])):
    #     train_data, valid_data = train.iloc[train_index], train.iloc[valid_index]
    #     print_info(f"Fold {i} Train", train_data)
    #     print_info(f"Fold {i} Valid", valid_data)
    #     train_data.to_json(f"train_fold_{i}.jsonl", orient="records", lines=True)
    #     valid_data.to_json(f"valid_fold_{i}.jsonl", orient="records", lines=True)


def divide_df_by_cwe(df: pd.DataFrame) -> None:
    print()
    target_df = df[df["target"] == 1]
    non_cwe_df = target_df[target_df["cwe_id"].apply(lambda x: x == [])]
    print(f"Non-CWE number: {non_cwe_df.shape[0]}, ratio: {non_cwe_df.shape[0] / target_df.shape[0]}")
    multi_cwe_df = target_df[target_df["cwe_id"].apply(lambda x: len(x) > 1)]
    print(f"Multi-CWE number: {multi_cwe_df.shape[0]}, ratio: {multi_cwe_df.shape[0] / target_df.shape[0]}")

    df = df[df["cwe_id"].apply(lambda x: x != [])]
    cwe_ids = df["cwe_id"].explode().unique()
    print(f"Total CWE: {len(cwe_ids)}")

    cwe_cnt = {}
    if not os.path.exists("./cwe_type"):
        os.mkdir("./cwe_type")
    for cwe_id in cwe_ids:
        if cwe_id == "":
            continue
        sub_df = df[df["cwe_id"].apply(lambda x: cwe_id in x)]
        cwe_cnt[cwe_id] = (
            sub_df.shape[0], sub_df[sub_df["target"] == 1].shape[0], sub_df[sub_df["target"] == 0].shape[0])
        # print_info(f"{cwe_id}", sub_df)
        if sub_df[sub_df["target"] == 1].shape[0] < 10:
            # print(f"Skip {cwe_id} because of less than 10 vul functions\n")
            continue

        # random split
        train, valid = train_test_split(sub_df, test_size=0.2, random_state=seed, stratify=sub_df["target"])
        valid, test = train_test_split(valid, test_size=0.5, random_state=seed, stratify=valid["target"])

        # print_info("Train", train)
        # print_info("Valid", valid)
        # print_info("Test", test)
        # print()

        train.to_json(f"./cwe_type/train_{cwe_id}.jsonl", orient="records", lines=True)
        valid.to_json(f"./cwe_type/valid_{cwe_id}.jsonl", orient="records", lines=True)
        test.to_json(f"./cwe_type/test_{cwe_id}.jsonl", orient="records", lines=True)

    topn = 8
    sorted_cwe = sorted(cwe_cnt.items(), key=lambda x: x[1][1], reverse=True)
    ratio = 2 ** 31
    print("Top 10 CWE:")
    with open("./cwe_type/log.csv", "w") as f:
        f.write(f"cwe_id,total,vul,non-vul,ratio\n")
        for cwe_id, (total, vul, non_vul) in sorted_cwe[:topn]:
            print(
                f"{cwe_id}: total: {total}, vul: {vul}, non-vul: {non_vul}, ratio: {non_vul / vul if vul != 0 else 0}")
            f.write(f"{cwe_id},{total},{vul},{non_vul},{non_vul / vul if vul != 0 else 0}\n")
            ratio = min(ratio, non_vul / vul if vul != 0 else 0)

    # 单独记录Top
    if not os.path.exists("./cwe_type/top"):
        os.mkdir("./cwe_type/top")
    for cwe_id, _ in sorted_cwe[:topn]:
        sub_df = df[df["cwe_id"].apply(lambda x: cwe_id in x)]
        # 从target=0中随机选取使得比例满足要求
        vul_df = sub_df[sub_df["target"] == 1]
        non_vul_df = sub_df[sub_df["target"] == 0]
        non_vul_df = non_vul_df.sample(n=int(ratio * vul_df.shape[0]), random_state=seed)
        sub_df = pd.concat([vul_df, non_vul_df], ignore_index=True)
        print_info(f"{cwe_id}", sub_df)

        # random split
        train, valid = train_test_split(sub_df, test_size=0.2, random_state=seed, stratify=sub_df["target"])
        valid, test = train_test_split(valid, test_size=0.5, random_state=seed, stratify=valid["target"])

        train.to_json(f"./cwe_type/top/train_{cwe_id}.jsonl", orient="records", lines=True)
        valid.to_json(f"./cwe_type/top/valid_{cwe_id}.jsonl", orient="records", lines=True)
        test.to_json(f"./cwe_type/top/test_{cwe_id}.jsonl", orient="records", lines=True)


def run(filename: str, cwe: bool, output_prefix: str = "") -> None:
    # df = pd.read_json(filename, lines=True)
    # print_info("Total", df)
    df = remove_non_vulnerable_record(filename)
    if cwe:
        divide_df_by_cwe(df)
    else:
        divide_df(df, output_prefix)


def test_truncated_num(filename: str) -> None:
    # 统计被截断的数量
    import json
    from transformers import (BertConfig, BertTokenizer, BertForSequenceClassification,
                              GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                              OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                              RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                              DistilBertConfig, DistilBertForSequenceClassification,
                              DistilBertTokenizer)
    MODEL_CLASSES = {
        'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
        'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
    tokenizer = tokenizer_class.from_pretrained("../EPVD/code/microsoft/codebert-base", do_lower_case=True)
    truncate_cnt = 0
    line_cnt = 0
    with open(filename) as f:
        for line in f:
            js = json.loads(line.strip())
            code = ' '.join(js['func'].split())
            code_tokens = tokenizer.tokenize(code)
            if len(code_tokens) > 400 - 2:
                truncate_cnt += 1
            line_cnt += 1
    print(f"Truncate count: {truncate_cnt}, ratio: {truncate_cnt / line_cnt}")


def get_vul_func_in_test():
    df = pd.read_json("11test.jsonl", lines=True)
    llm_df = pd.read_json("deepseek-ai/dataset_extracted_llm_1011.jsonl", lines=True)
    # vul_func = df[df["target"] == 1]
    vul_func = df
    vul_func["url"] = vul_func["url"] + "/commit/" + vul_func["commit_id"]
    vul_func["osv_id"] = "https://osv.dev/vulnerability/" + vul_func["osv_id"]
    vul_func = vul_func.rename(columns={"url": "repo_url", "osv_id": "osv_url"})
    vul_func["file_path"] = vul_func["file_path"]
    vul_func["func"] = vul_func["func"].apply(lambda x: x.split("\n")[0])

    # 给vul_func添加一列，内容为llm_df中hash相同的行的detail内容
    vul_func["detail"] = ""
    for idx, row in vul_func.iterrows():
        llm_row = llm_df[llm_df["hash"] == row["hash"]]
        if llm_row.shape[0] == 1:
            vul_func.at[idx, "detail"] = llm_row["detail"].values[0]
        else:
            assert llm_row.shape[0] == 0
            assert row["target"] == 0
    # write to xlsx file
    vul_func.to_excel("vul_func_in_test.xlsx",
                      columns=["idx", "osv_url", "repo_url", "cwe_id", "file_path", "func", "target", "detail"],
                      index=False)


def evaluate_by_cwe(prediction_file: str):
    from sklearn.metrics import recall_score, precision_score, f1_score
    predictions = {}
    with open(prediction_file) as f:
        for line in f:
            line = line.strip()
            idx, label = line.split()
            predictions[int(idx)] = int(label)
    answer_df = pd.read_json("11test.jsonl", lines=True)
    # 存储每个CWE的真实标签和预测标签
    cwe_metrics = {}
    for _, row in answer_df.iterrows():
        idx = row['idx']
        cwes = row['cwe_id']
        target = row['target']
        if idx not in predictions:
            raise ValueError(f"Missing prediction for index {idx}.")
        pred = predictions[idx]

        for cwe in cwes:
            if cwe not in cwe_metrics:
                cwe_metrics[cwe] = {'y_true': [], 'y_pred': []}
            cwe_metrics[cwe]['y_true'].append(target)
            cwe_metrics[cwe]['y_pred'].append(pred)

    # 计算每个CWE的评估指标
    results = {}
    for cwe, data in cwe_metrics.items():
        precision = precision_score(data['y_true'], data['y_pred'], zero_division=0)
        recall = recall_score(data['y_true'], data['y_pred'], zero_division=0)
        f1 = f1_score(data['y_true'], data['y_pred'], zero_division=0)

        results[cwe] = {
            'size': len(data['y_true']),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    sorted_results = sorted(results.items(), key=lambda x: x[0], reverse=False)

    # write to csv
    with open("cwe_metrics.csv", "w") as f:
        f.write("CWE,Size,Precision,Recall,F1 Score\n")
        for cwe, metrics in sorted_results:
            f.write(
                f"{cwe},{metrics['size']},{(metrics['precision'] * 100):.2f}%,{(metrics['recall'] * 100):.2f}%,{(metrics['f1_score'] * 100):.2f}%\n")


def get_top10_osv_id():
    df = pd.read_json("dataset_extracted.jsonl", lines=True)
    osv_count = {}
    for idx, row in df.iterrows():
        osv_id = row["osv_id"]
        if osv_id not in osv_count:
            osv_count[osv_id] = (0, 0, 0)
        if row["target"] == 1:
            osv_count[osv_id] = (osv_count[osv_id][0] + 1, osv_count[osv_id][1] + 1, osv_count[osv_id][2])
        else:
            osv_count[osv_id] = (osv_count[osv_id][0] + 1, osv_count[osv_id][1], osv_count[osv_id][2] + 1)
    sorted_osv = sorted(osv_count.items(), key=lambda x: x[1][0], reverse=True)
    print("Top 10 OSV:")
    for osv_id, (total, vul, non_vul) in sorted_osv[:10]:
        print(f"{osv_id}: total: {total}, vul: {vul}, non-vul: {non_vul}, ratio: {non_vul / vul if vul != 0 else 0}")


# 删除target均为0的commit_id（即该commit没有函数被标记为漏洞函数）
def remove_non_vulnerable_record(filename: str):
    df = pd.read_json(filename, lines=True)
    print_info("Before", df)
    valid_commit_ids = df[df["target"] == 1]["commit_id"].unique()
    filtered_df = df[df["commit_id"].isin(valid_commit_ids)]
    print_info("Filtered", filtered_df)
    # # 输出其中的osv_id种类
    # print("Total unique osv_id:", len(filtered_df["osv_id"].unique()))
    # # 输出其中的url种类
    # print("Total unique url:", len(filtered_df["url"].unique()))
    #
    # # 统计CWE种类以及每个CWE包含的osv_id数量
    # cwe_count = {}
    # for _, row in filtered_df.iterrows():
    #     for cwe in row["cwe_id"]:
    #         if cwe not in cwe_count:
    #             cwe_count[cwe] = set()
    #         cwe_count[cwe].add(row["osv_id"])
    # cwe_count = dict(sorted(cwe_count.items(), key=lambda x: len(x[1]), reverse=True))
    # print("CWE count:")
    # for i, (cwe, osv_ids) in enumerate(cwe_count.items()):
    #     if i % 5 == 0:
    #         print(f" \\\\\n{cwe} & {len(osv_ids)}", end="")
    #     else:
    #         print(f" & {cwe} & {len(osv_ids)}", end="")

    return filtered_df


if __name__ == "__main__":
    # run("dataset_extracted.jsonl", False)
    # run("dataset_extracted.jsonl", True)

    # run(f"ds_11100_relabeled_11d.jsonl", False, "ds11100")
    # for config in ["1111", "0111", "1011", "1101", "1110"]:
    #     print(f"config: {config}")
    #     run(f"deepseek-ai_11/dataset_extracted_llm_{config}_relabeled.jsonl", False, f"_{config}")

    run("explanation_11.jsonl", False, "e11")
    # run("dataset_extracted_explanation_o.jsonl", True)

    # remove_non_vulnerable_record("dataset_extracted_llm_11d_relabeled.jsonl")

    # test_truncated_num("dataset_extracted.jsonl")
    # get_vul_func_in_test()
    # evaluate_by_cwe("../predictions_e.txt")
    # get_top10_osv_id()
