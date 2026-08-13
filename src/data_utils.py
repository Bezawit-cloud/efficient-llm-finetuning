"""
data_utils.py — Alpaca dataset loading, formatting, and tokenisation helpers.
"""
from typing import Any, Dict, List, Optional
from src.utils import get_logger

logger = get_logger("data_utils")

# ── Alpaca prompt template ─────────────────────────────────────────────────
ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task"
    "{input_part}. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_section}"
    "### Response:\n{output}"
)


def format_alpaca_example(example: Dict[str, str], eos_token: str = "") -> str:
    """Format a single Alpaca example into a prompt string."""
    has_input = bool(example.get("input", "").strip())
    input_part = ", using the input below as context" if has_input else ""
    input_section = f"### Input:\n{example['input']}\n\n" if has_input else ""
    return ALPACA_PROMPT_TEMPLATE.format(
        input_part=input_part,
        instruction=example.get("instruction", ""),
        input_section=input_section,
        output=example.get("output", ""),
    ) + eos_token


def load_alpaca_dataset(config: Dict[str, Any]):
    """
    Load the Alpaca dataset from HuggingFace, apply max_samples cap,
    and return a train/eval split.
    """
    from datasets import load_dataset

    dataset_name = config["data"]["dataset_name"]
    max_samples = config["data"].get("max_samples", None)
    eval_ratio  = config["data"].get("eval_split_ratio", 0.05)
    seed        = config.get("seed", 42)

    logger.info(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")

    if max_samples is not None and max_samples < len(ds):
        logger.info(f"Capping at {max_samples} examples (from {len(ds)})")
        ds = ds.select(range(max_samples))

    # Train / eval split
    split = ds.train_test_split(test_size=eval_ratio, seed=seed)
    logger.info(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")
    return split["train"], split["test"]


def tokenize_dataset(
    examples_list: List[Dict[str, str]],
    tokenizer,
    max_seq_length: int = 512,
    eos_token: str = "",
):
    """
    Tokenise a list of example dicts into a HuggingFace Dataset ready for Trainer.
    Uses full-sequence training (instruction + response in one sequence).
    """
    from datasets import Dataset

    texts = [format_alpaca_example(ex, eos_token) for ex in examples_list]

    def tokenize_fn(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    raw_ds = Dataset.from_dict({"text": texts})
    tokenized_ds = raw_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenising",
    )
    return tokenized_ds


def dataset_from_list(examples_list: List[Dict[str, str]]):
    """Convert a plain Python list of dicts to a HuggingFace Dataset."""
    from datasets import Dataset
    return Dataset.from_list(examples_list)
