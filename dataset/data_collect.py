import json
import os
import re
import time
import urllib.parse
from enum import Enum

import requests
import tree_sitter_rust
from jsonpath_ng import parse
from tqdm import tqdm
from tree_sitter import Language, Parser

parser = Parser()
parser.set_language(Language(tree_sitter_rust.language(), "rust"))

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


class RepoType(Enum):
    GITHUB = 1
    GITLAB = 2


class VulInfo:
    def __init__(self, index: int, osv_id: str, cwe_id: list[str], summary: str, details: str, publish_date: str,
                 update_date: str, url: str, commit_id: str):
        self.index = index
        self.osv_id = osv_id
        self.cwe_id = cwe_id
        self.summary = summary
        self.details = details
        self.publish_date = publish_date
        self.update_date = update_date
        self.url = url
        self.commit_id = commit_id


class FileChange:
    def __init__(self, file_path: str, code_before: str, code_after: str):
        self.file_path = file_path
        self.code_before = code_before
        self.code_after = code_after

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "code_before": self.code_before,
            "code_after": self.code_after
        }


class CommitInfo:
    def __init__(self, repo_type: RepoType, prefix_url: str, repo_dir: str, commit_id: str, parent_commit_id: str,
                 commit_message: str, commit_date: str, diff: str, files: list[dict], files_changed: list[FileChange]):
        self.repo_type = repo_type
        self.prefix_url = prefix_url
        self.repo_dir = repo_dir
        self.commit_id = commit_id
        self.parent_commit_id = parent_commit_id
        self.commit_message = commit_message
        self.commit_date = commit_date
        self.diff = diff
        self.files = files
        self.files_changed = files_changed


def request_get(url: str, headers=None, timeout: int = 10) -> requests.Response:
    headers = headers or {}
    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0"
    # use GitHub token to avoid rate limit
    if "github.com" in url and GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    while True:
        response = None
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e:
            tqdm.write(f"Error requesting {url}: {e}")
            if response is not None and response.status_code < 500:
                exit(1)
            tqdm.write("Retrying in 5 seconds...")
            time.sleep(5)


def get_commit_data(vul_info: VulInfo) -> dict | None:
    # get commit and diff
    commit_info: CommitInfo
    if "github" in vul_info.url:
        repo_type = RepoType.GITHUB
        repo_dir = re.search(r"github.com/(.+)", vul_info.url).group(1)
        commit = request_get(f"https://api.github.com/repos/{repo_dir}/commits/{vul_info.commit_id}").json()
        diff = request_get(f"https://api.github.com/repos/{repo_dir}/commits/{vul_info.commit_id}",
                           headers={"Accept": "application/vnd.github.diff"}).text

        commit_info = CommitInfo(RepoType.GITHUB, "https://github.com/", repo_dir, commit["sha"],
                                 commit["parents"][0]["sha"], commit["commit"]["message"],
                                 commit["commit"]["author"]["date"], diff, commit["files"], [])
    elif "gitlab" in vul_info.url:
        repo_type = RepoType.GITLAB
        prefix_url = re.search(r"https://gitlab\.([^/]+)/", vul_info.url).group(0)
        repo_dir = re.search(r"https://gitlab\.([^/]+)/(.*)", vul_info.url).group(2)
        commit = request_get(
            f"{prefix_url}api/v4/projects/{urllib.parse.quote(repo_dir, safe="")}/repository/commits/{vul_info.commit_id}").json()
        diff = request_get(f"{prefix_url}{repo_dir}/-/commit/{vul_info.commit_id}.diff").text
        files = request_get(
            f"{prefix_url}api/v4/projects/{urllib.parse.quote(repo_dir, safe="")}/repository/commits/{vul_info.commit_id}/diff").json()

        commit_info = CommitInfo(RepoType.GITLAB, prefix_url, repo_dir, commit["id"], commit["parent_ids"][0],
                                 commit["message"], commit["authored_date"], diff, files, [])
    else:
        raise Exception(f"Unsupported repository: {vul_info.url}")

    # Get functions before and after the commit
    has_rust_file = False
    for i, file in enumerate(commit_info.files):
        match repo_type:
            case RepoType.GITHUB:
                filename = file["filename"]
            case RepoType.GITLAB:
                filename = file["new_path"]
            case _:
                raise Exception(f"Unsupported repository: {vul_info.url}")

        if not filename.endswith(".rs") or "doc" in filename or "test" in filename or "example" in filename:
            continue
        has_rust_file = True
        # get the code before and after the commit
        code_before, code_after = "", ""
        match repo_type:
            case RepoType.GITHUB:
                if file["status"] != "added":
                    if file["status"] == "renamed":
                        code_before = request_get(
                            f"https://raw.githubusercontent.com/{commit_info.repo_dir}/{commit_info.parent_commit_id}/{file['previous_filename']}").text
                    else:
                        code_before = request_get(
                            f"https://raw.githubusercontent.com/{commit_info.repo_dir}/{commit_info.parent_commit_id}/{file['filename']}").text
                if file["status"] != "removed":
                    code_after = request_get(file["raw_url"]).text
            case RepoType.GITLAB:
                if not file["new_file"]:
                    code_before = request_get(
                        f"{commit_info.prefix_url}{commit_info.repo_dir}/-/raw/{commit_info.parent_commit_id}/{file['old_path']}").text
                if not file["deleted_file"]:
                    code_after = request_get(
                        f"{commit_info.prefix_url}{commit_info.repo_dir}/-/raw/{commit_info.commit_id}/{file['new_path']}").text
            case _:
                raise Exception(f"Unsupported repository: {vul_info.url}")
        commit_info.files_changed.append(FileChange(filename, code_before, code_after))
    if not has_rust_file:
        return None

    commit_data = {
        "index": vul_info.index,
        "osv_id": vul_info.osv_id,
        "cwe_id": vul_info.cwe_id,
        "language": "Rust",
        "summary": vul_info.summary,
        "details": vul_info.details,
        "publish_date": vul_info.publish_date,
        "update_date": vul_info.update_date,
        "url": vul_info.url,
        "commit_id": commit_info.commit_id,
        "commit_message": commit_info.commit_message,
        "commit_date": commit_info.commit_date,
        "diff": commit_info.diff,
        "files_changed": [file_change.to_dict() for file_change in commit_info.files_changed],
    }
    return commit_data


def process_json_from_directory(directory: str, output_file: str) -> None:
    log_file = str(output_file).split(".")[0] + "_log.json"
    start = 0
    index = 0
    if os.path.exists(log_file):
        with open(log_file, "r") as log:
            log_data = json.load(log)
            start = log_data["start"]
            index = log_data["index"]
            tqdm.write(f"Start from {start}, index: {index}")

    with open(output_file, "a+", newline="", encoding="utf-8") as jsonfile:
        with tqdm(total=len(os.listdir(directory)), desc="Processing") as pbar:
            for i, filename in enumerate(os.listdir(directory)):
                pbar.set_description(f"Processing #{i} {filename}")
                if i < start:
                    pbar.update()
                    continue
                if not filename.endswith(".json"):
                    tqdm.write(f"# {i} {filename} is not a json file")
                    pbar.update()
                    continue
                # write the log
                if i > 0:
                    with open(log_file, "w") as log:
                        json.dump({"start": i, "index": index}, log, indent=4)

                with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                    data = json.load(file)

                # check if the vulnerability is fixed
                fixed = parse("$..fixed").find(data)
                if not fixed:
                    tqdm.write(f"# {i} {filename} is not fixed")
                    pbar.update()
                    continue

                # get the PR or commit url
                if "references" not in data:
                    tqdm.write(f"# {i} {filename} has no references url")
                    pbar.update()
                    continue
                urls = []
                web_urls = []
                for reference in data["references"]:
                    if reference["type"] == "WEB":
                        if len(urls) == 0 \
                                and not reference["url"].endswith(".md") \
                                and not re.match(r"https://rustsec\.org", reference["url"]) \
                                and not re.match(r"https://nvd\.nist\.gov", reference["url"]) \
                                and not re.match(r"https://github\.com/.+/.+/issues/\d+", reference["url"]) \
                                and not re.match(r"https://github\.com/.+/.+(/)?$", reference["url"]) \
                                and not re.match(r"https://gitlab\.([^/]+)/(.+)/-/issues/\d+", reference["url"]):
                            web_urls.append(reference["url"])

                        # github
                        if re.match(r"https://github\.com/([^/]+)/([^/]+)/(pull/\d+|commit/[0-9a-f]+)",
                                    reference["url"]):
                            urls.append(reference["url"])
                        # gitlab
                        if re.match(r"https://gitlab\.([^/]+)/(.+)/-/(merge_requests/\d+|commit/[0-9a-f]+)",
                                    reference["url"]):
                            urls.append(reference["url"])
                        # git.openssl.org
                        if re.match(r"https://git\.openssl\.org/gitweb/\?p=openssl\.git",
                                    reference["url"]):
                            openssl_hash = re.search(r"h=([0-9a-f]+)", reference["url"])
                            assert openssl_hash
                            urls.append(f"https://github.com/openssl/openssl/commit/{openssl_hash.group(1)}")
                if len(urls) == 0:
                    if len(web_urls) > 0:
                        tqdm.write(
                            f"# {i} {filename} has no PR or commit url, but has possible web urls: {web_urls}")
                    else:
                        tqdm.write(f"# {i} {filename} has no PR or commit url")
                    pbar.update()
                    continue

                commits_processed = []
                commit_datas = []
                for url in urls:
                    if re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/\d+", url):
                        # get the merge commit hash from the PR
                        pr_url = re.search(r"https://github\.com/([^/]+)/([^/]+)/pull/\d+", url).group(0)
                        pr_url = pr_url.replace("github.com", "api.github.com/repos").replace("pull", "pulls")
                        pr_url = re.sub(r"/commits/[^/]+", "", pr_url)
                        pr_data = request_get(pr_url).json()
                        if pr_data["state"] == "open":
                            tqdm.write(f"# {i} {filename} PR is open, skip it. PR url: {url}")
                            continue
                        if pr_data["merged"]:
                            commit_hash = pr_data["merge_commit_sha"]
                        else:
                            tqdm.write(
                                f"# {i} {filename} PR is not merged, please check it and input the commit hash manually. PR url: {url}")
                            exit(1)
                        repo_url = pr_data["base"]["repo"]["html_url"]
                    elif re.match(r"https://github\.com/([^/]+)/([^/]+)/commit/[0-9a-f]+", url):
                        repo_url = re.search(r"https://github\.com/([^/]+)/([^/]+)", url).group(0)
                        commit_hash = url.split("/")[-1]
                    elif re.match(r"https://gitlab\.([^/]+)/(.+)/-/(merge_requests/\d+)", url):
                        mr_url_group = re.search(r"https://gitlab\.([^/]+)/(.+)/-/(merge_requests/\d+)", url)
                        repo_url = f"https://gitlab.{mr_url_group.group(1)}/{mr_url_group.group(2)}"
                        mr_api_url = f"https://gitlab.{mr_url_group.group(1)}/api/v4/projects/{urllib.parse.quote(mr_url_group.group(2), safe="")}/merge_requests/{mr_url_group.group(3).split('/')[-1]}"
                        mr_data = request_get(mr_api_url).json()
                        if mr_data["state"] == "merged":
                            commit_hash = mr_data["merge_commit_sha"]
                        else:
                            tqdm.write(
                                f"# {i} {filename} MR is not merged, please check it and input the commit hash manually. MR url: {url}")
                            exit(1)
                    elif re.match(r"https://gitlab\.([^/]+)/(.+)/-/(commit/[0-9a-f]+)", url):
                        mr_url_group = re.search(r"https://gitlab\.([^/]+)/(.+)/-/(commit/[0-9a-f]+)", url)
                        repo_url = f"https://gitlab.{mr_url_group.group(1)}/{mr_url_group.group(2)}"
                        commit_hash = mr_url_group.group(3).split("/")[-1]
                    else:
                        tqdm.write(f"# {i} {filename} has an unsupported url, please check it manually. URL: {url}")
                        exit(1)
                    repo_url = repo_url.rstrip("/")

                    # check if the commit has been processed
                    processed = False
                    for commit in commits_processed:
                        if commit.startswith(commit_hash):
                            tqdm.write(f"# {i} {filename} {commit_hash} has been processed")
                            processed = True
                            break
                    if processed:
                        continue

                    cwe_ids = data["database_specific"]["cwe_ids"] if "cwe_ids" in data["database_specific"] else []
                    # get the commit data
                    vul_info = VulInfo(index, data["id"], cwe_ids, data["summary"],
                                       data["details"], data["published"], data["modified"], repo_url, commit_hash)
                    if (commit_data := get_commit_data(vul_info)) is not None:
                        commit_datas.append(commit_data)
                        commits_processed.append(commit_data["commit_id"])
                        index += 1
                    else:
                        commits_processed.append(commit_hash)
                        tqdm.write(f"# {i} {filename} {commit_hash} has no Rust file")

                for commit_data in commit_datas:
                    json.dump(commit_data, jsonfile, ensure_ascii=False)
                    jsonfile.write("\n")
                pbar.update()
        with open(log_file, "w") as log:
            json.dump({"start": i, "index": index}, log, indent=4)


if __name__ == "__main__":
    process_json_from_directory("osv", "dataset_osv.jsonl")
