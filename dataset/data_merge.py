import pandas as pd


def get_priority(cve_id: str) -> int:
    if cve_id.startswith("CVE"):
        return 1
    elif cve_id.startswith("GHSA"):
        return 2
    elif cve_id.startswith("RUSTSEC"):
        return 3
    return 4


def process_group(group: pd.DataFrame) -> pd.DataFrame:
    res = group.drop_duplicates(subset=["osv_id"], keep="first")
    ids = res["osv_id"].tolist()

    # 相同hash的id只保留一个
    if len(ids) > 1:
        res = res.iloc[[min(range(len(ids)), key=lambda i: get_priority(ids[i]))]]
    # merge all cwe_id
    cwe_id = set()
    for cwe in group["cwe_id"]:
        cwe_id.update(cwe)
    res["cwe_id"] = [list(cwe_id)]
    return res


if __name__ == "__main__":
    df = pd.read_json("dataset_osv.jsonl", lines=True)
    dfs = [df]
    merged_df = pd.concat(dfs, ignore_index=True)
    print("Before:", merged_df.shape)
    # remove duplicate content
    merged_df = merged_df.groupby(["commit_id"], group_keys=False).apply(process_group)
    merged_df = merged_df.sort_values(by=["index"])

    print("After:", merged_df.shape)
    merged_df.to_json("dataset_merged.jsonl", orient="records", lines=True)
