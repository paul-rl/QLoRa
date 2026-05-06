# QLoRa

Fine-tuning [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and comparing **NF4 vs FP4 quantization** using [MMLU](https://huggingface.co/datasets/cais/mmlu) and [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) benchmarks.

---

## Overview

This project uses QLoRA (Quantized Low-Rank Adaptation) to fine-tune Llama 3 8B Instruct on the UltraChat 200k dataset under three training configurations:

- **NF4** — 4-bit NormalFloat quantization (50k training samples)
- **FP4** — 4-bit FloatingPoint quantization (50k training samples)
- **Baseline** — Full bfloat16/fp16 (sanity-check run, 1 sample)

Each adapter is then evaluated on MMLU (accuracy) and MT-Bench (GPT-4 judge scoring).

---

## Repository Structure

```
QLoRa/
├── qlora_training.py           # Fine-tuning script (NF4, FP4, baseline)
├── mmlu_eval.py                # MMLU evaluation via lm-eval harness
├── mt_bench_eval_(3).ipynb     # MT-Bench evaluation notebook (Google Colab)
├── qlora_experiments/          # Output directory for saved adapters
├── mmlu_results/               # MMLU evaluation outputs
├── requirements.txt            # Python dependencies
└── LICENSE
```

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key packages include `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets`, `lm_eval`, and `torch` with CUDA 12.8 support. PyTorch is pulled from the extra index URL already specified in `requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.10.0+cu128
```

A CUDA-capable GPU is required for training.

---

## Environment Variables

The following environment variables / secrets must be set before running:

| Variable | Where used | Description |
|---|---|---|
| `HF_TOKEN` | `mmlu_eval.py` (Hugging Face hub access) | HuggingFace access token — required to download the gated `meta-llama/Meta-Llama-3-8B-Instruct` model and the UltraChat dataset |
| `OPENAI_API_KEY` | `mt_bench_eval_(3).ipynb` | OpenAI API key used by the MT-Bench GPT-4 judge |

Set them in your shell before running:

```bash
export HF_TOKEN="hf_..."
export OPENAI_API_KEY="sk-..."
```

Or, in Google Colab, add them via **Secrets** (the 🔑 panel in the left sidebar).

---

## Usage

### 1. Training

Runs all three configurations (NF4, FP4, baseline) sequentially and saves a `comparison_summary.json`. The model (`meta-llama/Meta-Llama-3-8B-Instruct`) and dataset (`HuggingFaceH4/ultrachat_200k`) are downloaded automatically from the Hugging Face Hub — make sure `HF_TOKEN` is set first.

```bash
python qlora_training.py
```

Training outputs (adapter weights + `results.json`) are saved under `qlora_experiments_extra/`.

### 2. MMLU Evaluation

Evaluates all three saved adapters on the `mmlu_high_school_computer_science` task (5-shot):

```bash
python mmlu_eval.py
```

Results are written to `mmlu_results/`. The evaluated task can be changed by editing the `TASK` variable at the top of `mmlu_eval.py`.

### 3. MT-Bench Evaluation

> ⚠️ **Google Colab required.** The MT-Bench evaluation notebook (`mt_bench_eval_(3).ipynb`) is designed to run in a Google Colab environment. It relies on Colab-specific utilities for mounting Google Drive, installing dependencies inline, and accessing Colab secrets. Running it locally may require manual adaptation.

Open the notebook directly in Colab, ensure `HF_TOKEN` and `OPENAI_API_KEY` are set in Colab Secrets, and run all cells.

---

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Base model | Meta-Llama-3-8B-Instruct |
| Dataset | UltraChat 200k (train_sft split) |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Max sequence length | 1024 |
| Batch size | 4 |
| Gradient accumulation steps | 2 |
| Learning rate | 1e-4 |
| Epochs | 1 |
| Double quantization | Enabled |
| Seed | 42 |

---

## License

MIT
