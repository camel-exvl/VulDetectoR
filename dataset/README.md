Rust漏洞函数数据集构建包含以下几个步骤：

1. **数据收集**：从[osv](https://osv-vulnerabilities.storage.googleapis.com/crates.io/all.zip)下载漏洞数据库，解压至`osv`
   目录下，使用`data_collect.py`构建漏洞信息数据集，并使用`data_merge.py`完成漏洞的去重与合并；
2. **函数提取**：使用`data_extract.py`提取漏洞代码函数并完成初步标注；
3. **函数标注**：使用`data_label_llm.py`使用大模型对函数进行标注，生成漏洞函数数据集。此时需配置环境变量`KEY`为大模型API密钥，
   `URL`为大模型API地址，同时传入参数指定使用的模型名称。