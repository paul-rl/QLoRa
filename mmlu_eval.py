import subprocess
import sys
from pathlib import Path


BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

TASK = "mmlu_high_school_computer_science"
NUM_FEWSHOT = 5
BATCH_SIZE = 1
DTYPE = "float32"

ADAPTERS = {
    "baseline": "qlora_experiments/llama3_8b_baseline/final_adapter",
    "fp4": "qlora_experiments/llama3_8b_fp4/final_adapter",
    "nf4": "qlora_experiments/llama3_8b_nf4/final_adapter",
}


def run_eval(model_name: str, adapter_path: str):
    output_path = Path(f"./mmlu_results/{model_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adapter_path = str(Path(adapter_path).resolve())

    model_args = (
        f"pretrained={BASE_MODEL},"
        f"peft={adapter_path},"
        f"dtype={DTYPE}"
    )

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", TASK,
        "--num_fewshot", str(NUM_FEWSHOT),
        "--batch_size", str(BATCH_SIZE),
        "--output_path", str(output_path),
    ]

    print("\nRunning evaluation for:", model_name)
    print("Command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    for model_name, adapter_path in ADAPTERS.items():
        run_eval(model_name, adapter_path)

    print("\nAll evaluations completed.")