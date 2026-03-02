结合大语言模型来识别Rust代码中存在的漏洞，并生成解释型文本。

# 文件结构

```
.
├── code
│   ├── accuracy.py # 计算准确率
│   ├── eval_all.py # 模型推理
│   ├── explanation_evaluation.py # 评估漏洞解释文本
│   ├── train_all.py # 模型训练
│   └── run.sh # 运行脚本
├── dataset
│   ├── e11dutrain.jsonl # 训练集
│   ├── e11duvalid.jsonl # 验证集
│   └── e11dutest.jsonl # 测试集
├── CodeLlama-13b-Instruct-hf # 预训练模型
└── README.md # 项目说明
```

# 数据集

数据集位于`dataset`目录下，从[OSV](https://osv.dev/)收集而来，根据漏洞信息中的commit id获取对应的代码。数据截至2025年2月。

数据集格式：
```json
{
  "func": "<Rust函数代码>",
  "hash": "<代码哈希值>",
  "url": "<仓库链接>",
  "commit_id": "<提交ID>",
  "commit_date": "<提交日期>",
  "message": "<提交信息>",
  "Explanation": "<漏洞解释文本>",
  "target": "<是否存在漏洞>",
  "idx": "<数据索引>"
}
```

# 运行脚本

在`code`目录下运行`run.sh`脚本来使用模型。

运行要求：
- Python 3.8.10
- Python 依赖: `pip install -r requirements.txt`
- [CodeLlama-13b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-13b-Instruct-hf)模型置于`./CodeLlama-13b-Instruct-hf`目录下

模型推理：
```bash
./run.sh eval [GPU_ID]
```
模型训练：

```bash
./run.sh train [GPU_ID]
```