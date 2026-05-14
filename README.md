# VulDetectoR

VulDetectoR is a tool designed to detect and explain vulnerabilities in Rust code using large language models. 

## Key Features

- Rust vulnerability classification at function level
- Explanation generation for model decisions

## Requirements

- Python 3.8.10
- Python dependencies: `pip install -r requirements.txt`
- Pre-trained model: [CodeLlama-13b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-13b-Instruct-hf)

## Quick Start

1. Clone the repository and navigate to the `code` directory.
2. Install the required dependencies and download the pre-trained model.
3. Run `vuldetector.py` to perform vulnerability detection and explanation generation.
```bash
python vuldetector.py --code_path [CODE_PATH] --model_path [MODEL_PATH]
```

## Dataset
The dataset is located in the `dataset` directory and is collected from [OSV](https://osv.dev/). It includes Rust code snippets with associated vulnerability information, commit details, and explanations.

The dataset is used for fine-tuning and testing the model's performance, and the format is as follows:
```json
{
  "func": "<Rust function code>",
  "hash": "<Code hash value>",
  "url": "<Repository link>",
  "commit_id": "<Commit ID>",
  "commit_date": "<Commit date>",
  "message": "<Commit message>",
  "Explanation": "<Vulnerability explanation text>",
  "target": "<Whether there is a vulnerability>",
  "idx": "<Data index>"
}
```

## Fine-tuning
To fine-tune the model on the dataset, run the following command:
```bash
python train_all.py --model_path [MODEL_PATH]
```

To evaluate the model's performance on the test set, run:
```bash
python eval_all.py --model_path [MODEL_PATH]
python accuracy.py
```