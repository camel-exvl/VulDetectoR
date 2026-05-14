import difflib
import hashlib
import re

import pandas as pd
import tree_sitter_rust
from tqdm import tqdm
from tree_sitter import Language, Parser, Node

parser = Parser()
parser.set_language(Language(tree_sitter_rust.language(), "rust"))


def get_function(code: str) -> list:
    code = re.sub(r"(?<!r#)\btry!", r"r#try!", code)
    tree = parser.parse(code.encode("utf-8"))

    def dfs(cur: Node) -> list:
        if cur.type == "mod_item":  # ignore test module
            identifier = cur.children[-2]
            assert identifier.type == "identifier"
            if b"test" in identifier.text:
                return []
        ret: list[Node] = []
        for node in cur.children:
            if node.type == "function_item":
                # ignore #[test] functions
                if node.prev_sibling is not None and node.prev_sibling.type == "attribute_item" and b"test" in node.prev_sibling.text:
                    continue
                if node.text is not None:
                    ret.append(node)
            elif node.children:
                ret.extend(dfs(node))
        return ret

    return dfs(tree.root_node)


def get_called_func(func: Node) -> set:
    called_func: set[str] = set()
    for node in func.children:
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node is not None and func_node.type == "identifier":
                called_func.add(func_node.text.decode("utf-8"))
        elif node.children:
            called_func.update(get_called_func(node))
    return called_func


def extract_function(row: pd.Series, output_df: pd.DataFrame) -> pd.DataFrame:
    vul_func: list[tuple[Node, list[str], str]] = []  # (Node, context_func_text, file_path)
    non_vul_func: list[tuple[Node, list[str], str]] = []
    for file in row["files_changed"]:
        func_before: list[Node] = get_function(file["code_before"])
        func_before: dict[Node, None] = {func: None for func in func_before}
        func_after: list[Node] = get_function(file["code_after"])
        func_after: dict[Node, None] = {func: None for func in func_after}
        # for func in get_function(file["code_after"]):
        #     non_vul_func.append((func, [], file["file_path"]))

        # get diff line numbers
        diff = list(
            difflib.unified_diff(file["code_before"].splitlines(), file["code_after"].splitlines(), lineterm="", n=0))
        for line in diff:
            if line.startswith("@@"):
                # Attention: diff信息中的行号是从1开始的，而tree-sitter中的行号是从0开始的
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                line_start_before = int(match.group(1)) - 1
                line_end_before = line_start_before + int(match.group(2) or 1) - 1
                line_start_after = int(match.group(3)) - 1
                line_end_after = line_start_after + int(match.group(4) or 1) - 1
                # 不管是增加还是删除，只要涉及到的行号在函数内，就认为是漏洞函数
                for func in list(func_before.keys()):
                    if (
                            func.start_point[0] <= line_start_before <= func.end_point[0]
                            or func.start_point[0] <= line_end_before <= func.end_point[0]
                            or line_start_before <= func.start_point[0] and line_end_before >= func.end_point[0]):
                        called_func = get_called_func(func)
                        context_func = []
                        for other_func in func_before:
                            func_node = other_func.child_by_field_name("name")
                            if func_node is not None:
                                func_name = func_node.text.decode("utf-8")
                                if func_name in called_func:
                                    context_func.append(other_func.text.decode("utf-8"))
                        vul_func.append((func, context_func, file["file_path"]))
                        func_before.pop(func)
                # 只记录发生修改的函数，将修复后的版本作为无漏洞版本
                for func in list(func_after.keys()):
                    if (
                            func.start_point[0] <= line_start_after <= func.end_point[0]
                            or func.start_point[0] <= line_end_after <= func.end_point[0]
                            or line_start_after <= func.start_point[0] and line_end_after >= func.end_point[0]):
                        called_func = get_called_func(func)
                        context_func = []
                        for other_func in func_after:
                            func_node = other_func.child_by_field_name("name")
                            if func_node is not None:
                                func_name = func_node.text.decode("utf-8")
                                if func_name in called_func:
                                    context_func.append(other_func.text.decode("utf-8"))
                        non_vul_func.append((func, context_func, file["file_path"]))
                        func_after.pop(func)
        # for func in func_before:
        #     non_vul_func.append((func, [], file["file_path"]))

    new_rows = []
    for func, context_funcs, path in vul_func:
        new_rows.append(
            [func.text.decode("utf-8"), 1, hashlib.sha256(func.text).hexdigest(), row["osv_id"], row["cwe_id"],
             row["url"], row["commit_id"],
             row["commit_date"], path])
    for func, context_funcs, path in non_vul_func:
        new_rows.append(
            [func.text.decode("utf-8"), 0, hashlib.sha256(func.text).hexdigest(), row["osv_id"], row["cwe_id"],
             row["url"], row["commit_id"],
             row["commit_date"], path])

    new_df = pd.DataFrame(new_rows,
                          columns=["func", "target", "hash", "osv_id", "cwe_id", "url", "commit_id", "commit_date",
                                   "file_path"])
    output_df = pd.concat([output_df, new_df], ignore_index=True)
    return output_df

    # tqdm.write(f"osv_id: {row['osv_id']}, vul: {len(vul_func)}, non-vul: {len(non_vul_func)}")


def run(filename: str, output: str) -> None:
    input_df = pd.read_json(filename, lines=True)
    print("Before:", input_df.shape)
    output_df = pd.DataFrame(
        columns=["func", "target", "hash", "osv_id", "cwe_id", "url", "commit_id", "commit_date", "file_path"])
    for index, row in tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Extracting", unit="row"):
        if row["osv_id"] == "GHSA-x4mq-m75f-mx8m":  # 这条记录有七万多个函数
            continue
        output_df = extract_function(row, output_df)

    output_df = output_df.drop_duplicates(subset=["hash"], keep="first")
    output_df["idx"] = output_df.index
    print("After:", output_df.shape)
    vul_cnt = output_df[output_df["target"] == 1].shape[0]
    non_vul_cnt = output_df[output_df["target"] == 0].shape[0]
    print("Vul:", vul_cnt, "Non-Vul:", non_vul_cnt, "ratio:", non_vul_cnt / vul_cnt)
    output_df.to_json(output, orient="records", lines=True)


if __name__ == "__main__":
    run("dataset_merged.jsonl", "dataset_extracted_11.jsonl")
