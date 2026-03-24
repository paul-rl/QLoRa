import os
import gc
import json
import time
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

model_name = "./Meta-Llama-3-8B-Instruct"
dataset_name = "./ultrachat_200k"
output_root = "qlora_experiments_extra"

seed = 42
max_length = 1024
batch_size = 4
grad_accum = 2
learning_rate = 1e-4
num_train_epochs = 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def bytes_to_gb(x):
    return x / (1024 ** 3)


def get_memory():
    return {
        "allocated_gb": bytes_to_gb(torch.cuda.memory_allocated()),
        "reserved_gb": bytes_to_gb(torch.cuda.memory_reserved()),
        "peak_allocated_gb": bytes_to_gb(torch.cuda.max_memory_allocated()),
        "peak_reserved_gb": bytes_to_gb(torch.cuda.max_memory_reserved()),
    }

def build_model(mode):
    """
    modes: {"nf4", "fp4", "baseline"}
    """
    bf16_supported = torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.bfloat16 if bf16_supported else torch.float16

    if mode in ["nf4", "fp4"]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_quant_type=mode,
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            dtype=model_dtype,
            device_map={"": 0},
            local_files_only=True
        )

        model = prepare_model_for_kbit_training(model)
    elif mode == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
            device_map={"": 0},
            local_files_only=True
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

def load_data():
    dataset = load_dataset(dataset_name, split="train_sft")

    def has_final_assistant(example):
        msgs = example["messages"]
        return bool(msgs) and msgs[-1]["role"] == "assistant"
    
    dataset = dataset.filter(has_final_assistant)

    def to_prompt_completion(example):
        msgs = example["messages"]
        return {
            "prompt": msgs[:-1],
            "completion": [msgs[-1]],
        }

    dataset = dataset.map(
        to_prompt_completion,
        remove_columns=dataset.column_names,
    )

    return dataset

def run_experiment(mode):
    run_dir = os.path.join(output_root, f"llama3_8b_{mode}")
    final_dir = os.path.join(run_dir, "final_adapter")
    os.makedirs(run_dir, exist_ok=True)

    cleanup()
    set_seed(seed)

    memory = {}

    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = build_model(mode)
    memory["after_model_load"] = get_memory()

    if mode in ["nf4", "fp4"]:
        train_dataset = load_data().select(range(50000))
    elif mode == "baseline":
        #small sanity check to demonstrate memory
        train_dataset = load_data().select(range(1))


    bf16_supported = torch.cuda.is_bf16_supported()

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=run_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            logging_steps=10,
            eval_strategy="no",
            save_strategy="no",
            gradient_checkpointing=True,
            bf16=bf16_supported,
            fp16=not bf16_supported,
            report_to="none",
            seed=seed,
            max_length=max_length,
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer
    )

    torch.cuda.reset_peak_memory_stats()
    memory["before_train"] = get_memory()

    start_time = time.time()

    train_result = trainer.train()

    train_time = time.time() - start_time
    memory["after_train"] = get_memory()

    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    results = {
        "mode": mode,
        "train_time_sec": train_time,
        "memory": memory,
        "trainer_metrics": train_result.metrics
    }

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    del trainer, model, tokenizer, train_dataset
    cleanup()

    return results

if __name__ == "__main__":
    os.makedirs(output_root, exist_ok=True)

    nf4_results = run_experiment("nf4")
    fp4_results = run_experiment("fp4")
    baseline_results = run_experiment("baseline")

    summary = {
        "nf4": nf4_results,
        "fp4": fp4_results,
        "baseline": baseline_results
    }

    with open(os.path.join(output_root, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
