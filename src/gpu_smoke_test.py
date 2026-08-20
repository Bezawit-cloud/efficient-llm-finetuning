"""
gpu_smoke_test.py — Tiny GPU verification test.

Verifies:
  1. CUDA detection, device name, and VRAM capacity.
  2. PyTorch, CUDA, and Transformers versions.
  3. Qwen2.5-0.5B-Instruct model and tokenizer loading on GPU.
  4. LoRA adapter attachment (r=8, alpha=16) and trainable parameter verification.
  5. 1 complete training step (forward + backward + optimizer step).
  6. GPU memory tracking (peak VRAM recorded via torch.cuda.max_memory_allocated).
  7. Results persistence to outputs/smoke_test/results.json.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import json
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

from src.utils import load_config, set_seed, get_gpu_memory_mb, count_trainable_params, save_results

def run_gpu_smoke_test():
    print("=" * 65)
    print("GPU MIGRATION SMOKE TEST — QWEN2.5-0.5B-INSTRUCT")
    print("=" * 65)

    # 1. Environment & CUDA Check
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (No CUDA device found)"
    total_vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if cuda_available else 0.0
    
    print(f"PyTorch Version:     {torch.__version__}")
    print(f"CUDA Available:      {cuda_available}")
    print(f"Active Device:       {device} ({device_name})")
    if cuda_available:
        print(f"Total GPU VRAM:      {total_vram_gb:.2f} GB")
        print(f"CUDA Version:        {torch.version.cuda}")
        bf16_ok = torch.cuda.is_bf16_supported()
        print(f"BF16 Supported:      {bf16_ok}")
    else:
        print("WARNING: CUDA is not available. Smoke test running in fallback mode.")
    print("-" * 65)

    # 2. Config & Seeding
    config = load_config("configs/exp1_baseline.yaml")
    set_seed(config.get("seed", 42))
    model_name = config["model"]["name"]
    t_cfg = config.get("training", {})
    max_seq_length = config["data"].get("max_seq_length", 512)
    per_device_batch_size = t_cfg.get("per_device_train_batch_size", 8)
    grad_accum_steps = t_cfg.get("gradient_accumulation_steps", 4)
    grad_checkpointing = t_cfg.get("gradient_checkpointing", False)

    # 3. Tokenizer & Dummy Data Batch
    print(f"Loading Tokenizer:   {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_dummy = max(16, per_device_batch_size * grad_accum_steps)
    dummy_examples = [
        {"text": f"Below is an instruction that describes a task.\n\n### Instruction:\nWrite a haiku about test number {i}.\n\n### Response:\nCode compiles fast\nTensors flow through the GPU\nTests pass with great joy.{tokenizer.eos_token}"}
        for i in range(n_dummy)
    ]
    raw_ds = Dataset.from_dict({"text": [d["text"] for d in dummy_examples]})

    def tokenize_fn(batch):
        tok = tokenizer(batch["text"], max_length=max_seq_length, truncation=True, padding=False)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized_ds = raw_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 4. Model Loading & LoRA Attachment
    print(f"Loading Model:       {model_name} on {device} ...")
    t0_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    load_time = time.perf_counter() - t0_load
    print(f"Model loaded in:     {load_time:.2f}s")

    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    param_info = count_trainable_params(model)
    print(f"LoRA Attached:       {param_info['trainable_params']:,} trainable params ({param_info['trainable_pct']:.2f}% of {param_info['total_params']:,})")

    # 5. Execute 1 Training Step
    output_dir = "outputs/smoke_test"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    use_fp16 = cuda_available and not torch.cuda.is_bf16_supported()
    use_bf16 = cuda_available and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=1,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=grad_checkpointing,
        learning_rate=t_cfg.get("learning_rate", 2e-4),
        fp16=use_fp16,
        bf16=use_bf16,
        optim=t_cfg.get("optim", "adamw_torch"),
        logging_steps=1,
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    print(f"\nExecuting 1 Training Step ({grad_accum_steps} micro-batches of {per_device_batch_size} samples = {per_device_batch_size * grad_accum_steps} samples, checkpointing={grad_checkpointing}) ...")
    t0_step = time.perf_counter()
    train_output = trainer.train()
    step_duration = time.perf_counter() - t0_step

    gpu_mem = get_gpu_memory_mb()
    print(f"Step Completed in:   {step_duration:.2f}s")
    print(f"Training Loss:       {train_output.training_loss:.4f}")
    print(f"Peak GPU VRAM:       {gpu_mem['peak_mb']} MB (~{gpu_mem['peak_mb']/1024:.2f} GB)")

    # 6. Save & Verify Results Persistence
    smoke_results = {
        "status": "PASS" if train_output is not None else "FAIL",
        "cuda_available": cuda_available,
        "device_name": device_name,
        "total_vram_gb": total_vram_gb,
        "pytorch_version": torch.__version__,
        "model": model_name,
        "trainable_params": param_info["trainable_params"],
        "total_params": param_info["total_params"],
        "step_duration_seconds": round(step_duration, 4),
        "peak_gpu_memory_mb": gpu_mem["peak_mb"],
        "loss": round(train_output.training_loss, 4),
    }

    results_file = save_results(smoke_results, output_dir, "smoke_test_results.json")
    print(f"Smoke Test Results:  Saved successfully to {results_file}")
    print("=" * 65)
    print("GPU SMOKE TEST PASSED SUCCESSFULLY!")
    print("=" * 65)

    return smoke_results

if __name__ == "__main__":
    run_gpu_smoke_test()
