# Qwen3-VL-LoRA for Chest X-Ray Interpretation

This repository provides a straightforward implementation for fine-tuning the **Qwen3-VL-4B-Instruct** vision-language model using **LoRA** (Low-Rank Adaptation) to generate concise clinical impressions from chest X-ray images.

## Overview

The project aims to assist radiologists by automatically generating structured clinical impressions from MIMIC-CXR chest X-rays. It uses efficient fine-tuning techniques (LoRA) to adapt the large VLM with minimal computational overhead.

## Key Features

- **Model:** Fine-tunes [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
- **Technique:** Low-Rank Adaptation (LoRA) via the `peft` library.
- **Dataset:** Utilizes the [MIMIC-CXR dataset](https://huggingface.co/datasets/itsanmolgupta/mimic-cxr-dataset).
- **Smart Filtering:** Implements frequency capping for common "normal" impressions to prevent model bias and overfitting on repetitive phrases.
- **Evaluation:** Integrated evaluation suite using ROUGE and BLEU scores.
- **Tracking:** Weights & Biases (WandB) integration for experiment logging.
- **Cluster Ready:** Includes a SLURM-compatible shell script for distributed/cluster environments.

## Setup & Installation

### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (NVIDIA Volta/Ampere or newer recommended)
- `pip` or `conda`

### Installation
```bash
pip install transformers torch datasets pillow trl peft wandb evaluate rouge_score
```

## Usage

### Local Execution
You can run the fine-tuning script directly with custom hyperparameters:

```bash
python qwen3-vl-lora.py \
    --r 32 \
    --lr 1e-4 \
    --epochs 3 \
    --batch_size 2 \
    --max_samples 13
```

### SLURM Cluster Execution
A `run_finetune.sh` script is provided for cluster environments. Adjust the paths and partition details as needed:

```bash
sbatch run_finetune.sh
```

## Dataset Configuration

The script automatically downloads and processes the MIMIC-CXR dataset.
- **Filtering:** Removes examples without impressions.
- **Capping:** Limits exact matches of clinical impressions (default max 13) to ensure a diverse training set.
- **Splitting:** 80% Train, 10% Validation, 10% Test.

## Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--r` | 32 | LoRA rank |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 2 | Per-device training batch size |
| `--gradient_steps` | 2 | Gradient accumulation steps |
| `--warmup_ratio` | 0.1 | Warmup ratio |
| `--lora_dropout` | 0.05 | LoRA dropout |
| `--weight_decay` | 0.01 | Weight decay |
| `--max_samples` | 13 | Max samples per unique impression (deduplication) |
| `--eval_steps` | 200 | Eval and save frequency |
| `--bf16` | False* | Use bf16 precision (*defaulted to True if not specified) |
| `--fp16` | False | Use fp16 precision |
| `--early_stopping` | 3 | Early stopping patience |
| `--model_name` | Qwen/Qwen3-VL-4B-Instruct | Base VLM model |
| `--dataset` | itsanmolgupta/mimic-cxr-dataset | Hugging Face dataset |
| `--output_dir` | None | Override checkpoint output directory |

## Evaluation

After training, the script evaluates the model on the test set and reports:
- **ROUGE-1 / ROUGE-2 / ROUGE-L**
- **BLEU Score**

It also prints 5 random sample predictions comparing Ground Truth vs. Model Prediction.
