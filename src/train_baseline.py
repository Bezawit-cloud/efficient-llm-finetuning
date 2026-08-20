"""
train_baseline.py — unified training entry-point for all experiments (E1–E5 + ablations).

Usage
-----
    python src/train_baseline.py --config configs/exp1_baseline.yaml
    python src/train_baseline.py --config configs/exp4_adaptive50_curriculum.yaml --seed 123

The script:
  1. Loads & merges config
  2. Loads Alpaca, scores it (or loads cached scores)
  3. Selects + orders the subset according to experiment config
  4. Tokenises
  5. Loads model + applies LoRA via PEFT
  6. Trains with HuggingFace Trainer
  7. Saves checkpoint + results JSON (wall-clock, peak GPU mem, param counts, eval loss)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# Make sure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, set_seed, get_logger, get_gpu_memory_mb, \
                      get_system_ram_mb, count_trainable_params, save_results, Timer

logger = get_logger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Model + tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(config):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType

    model_name = config["model"]["name"]
    if config["model"].get("use_fallback", False):
        model_name = config["model"]["fallback"]
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
    )

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
    model.print_trainable_parameters()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Scoring cache
# ─────────────────────────────────────────────────────────────────────────────

def get_or_compute_scores(train_examples, config):
    """Load cached scores if available, else compute and cache them."""
    cache_path = Path("data") / "scored_alpaca.json"
    if cache_path.exists():
        logger.info(f"Loading cached scores from {cache_path}")
        with open(cache_path) as f:
            scored = json.load(f)
        if len(scored) >= len(train_examples):
            return scored[:len(train_examples)]

    logger.info("Computing importance scores (this runs once and is cached) ...")
    from src.scoring import score_dataset
    scored = score_dataset(train_examples, config)

    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(scored, f)
    logger.info(f"Scores cached to {cache_path}")
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train experiment")
    parser.add_argument("--config", required=True, help="Experiment config YAML path")
    parser.add_argument("--seed",   type=int,  default=None, help="Override seed")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run one step only (runtime sanity check for Day 1)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    exp_id   = config.get("experiment", {}).get("id", "EX")
    exp_name = config.get("experiment", {}).get("name", "unknown")
    seed     = config["seed"]
    set_seed(seed)

    logger.info(f"=== Experiment {exp_id}: {exp_name} | seed={seed} ===")
    logger.info(f"Config: {args.config}")

    # ── Flush any leftover GPU allocations from a previous experiment ─────────
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"GPU free at experiment start: "
                    f"{torch.cuda.mem_get_info()[0]/1024**3:.2f} GB / "
                    f"{torch.cuda.mem_get_info()[1]/1024**3:.2f} GB")

    # ── Load data ─────────────────────────────────────────────────────────────
    from src.data_utils import load_alpaca_dataset, tokenize_dataset
    train_ds_raw, eval_ds_raw = load_alpaca_dataset(config)

    # Convert HF dataset to list of dicts for scoring
    train_examples = [dict(ex) for ex in train_ds_raw]
    eval_examples  = [dict(ex) for ex in eval_ds_raw]

    if args.dry_run:
        logger.info("DRY RUN — capping at 100 examples for timing test")
        train_examples = train_examples[:100]
        eval_examples  = eval_examples[:20]
        config["training"]["num_train_epochs"] = 1

    # ── Score + select + order ────────────────────────────────────────────────
    scored = get_or_compute_scores(train_examples, config)
    from src.select_and_order import select_and_order
    selected = select_and_order(scored, config)
    logger.info(f"Training on {len(selected)} examples")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(config)
    param_info = count_trainable_params(model)
    logger.info(f"Trainable params: {param_info['trainable_params']:,} "
                f"({param_info['trainable_pct']:.2f}% of {param_info['total_params']:,})")

    # ── Tokenise ──────────────────────────────────────────────────────────────
    max_seq = config["data"]["max_seq_length"]
    eos = tokenizer.eos_token or ""
    train_tokenized = tokenize_dataset(selected,      tokenizer, max_seq, eos)
    eval_tokenized  = tokenize_dataset(eval_examples, tokenizer, max_seq, eos)

    # ── Trainer ───────────────────────────────────────────────────────────────
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
    t_cfg = config["training"]
    output_dir = t_cfg.get("output_dir", f"outputs/{exp_id}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_steps=t_cfg["warmup_steps"],    # transformers v5: float < 1.0 acts as ratio
        weight_decay=t_cfg["weight_decay"],
        fp16=t_cfg.get("fp16", False),
        bf16=t_cfg.get("bf16", False),
        optim=t_cfg.get("optim", "adamw_torch"),
        save_strategy=t_cfg.get("save_strategy", "epoch"),
        eval_strategy=t_cfg.get("eval_strategy", "epoch"),
        logging_steps=t_cfg.get("logging_steps", 50),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", False),
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 0),
        report_to="none",   # no external logging; results saved to JSON
        seed=seed,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True,
                                           label_pad_token_id=-100)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    logger.info("Training started ...")
    train_result = trainer.train()
    wall_clock = time.perf_counter() - t_start
    logger.info(f"Training done in {wall_clock:.1f}s ({wall_clock/60:.1f} min)")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("Running final evaluation ...")
    eval_metrics = trainer.evaluate()
    logger.info(f"Eval metrics: {eval_metrics}")

    # ── Save checkpoint + results ─────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    gpu_mem = get_gpu_memory_mb()
    ram     = get_system_ram_mb()

    results = {
        "experiment_id":   exp_id,
        "experiment_name": exp_name,
        "config_path":     args.config,
        "seed":            seed,
        "n_train_examples": len(selected),
        "n_eval_examples":  len(eval_examples),
        "data_fraction":   config["data"]["selection"].get("fraction", 1.0),
        "selection_method": config["data"]["selection"].get("method", "full"),
        "ordering":         config["data"].get("ordering", "random"),
        # Efficiency
        "wall_clock_seconds": round(wall_clock, 2),
        "wall_clock_minutes": round(wall_clock / 60, 2),
        "peak_gpu_memory_mb": gpu_mem.get("peak_mb", 0.0),
        "system_ram":         ram,
        # Model
        **param_info,
        # Quality
        **eval_metrics,
        # Training loss
        "train_loss": train_result.training_loss,
    }

    results_path = save_results(results, output_dir)
    logger.info(f"Results saved to {results_path}")
    logger.info("-" * 60)
    for k, v in results.items():
        logger.info(f"  {k}: {v}")
    logger.info("-" * 60)

    # ── Free GPU memory before the next experiment ─────────────────────────────
    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared after experiment.")


if __name__ == "__main__":
    main()
