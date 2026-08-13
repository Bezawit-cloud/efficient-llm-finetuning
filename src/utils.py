"""
utils.py — shared utilities: config loading, seeding, logging, timing.
"""
import os
import json
import random
import time
import logging
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(exp_config_path: str, base_config_path: str = "configs/base_config.yaml") -> Dict[str, Any]:
    """
    Deep-merge base_config.yaml with an experiment-specific config.
    Experiment values override base values.
    """
    def deep_merge(base: dict, override: dict) -> dict:
        merged = base.copy()
        for k, v in override.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    with open(base_config_path, "r") as f:
        base = yaml.safe_load(f)
    with open(exp_config_path, "r") as f:
        exp = yaml.safe_load(f)

    return deep_merge(base, exp)


def flatten_config(config: dict, prefix: str = "") -> dict:
    """Flatten nested config dict for logging / W&B."""
    flat = {}
    for k, v in config.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_config(v, full_key))
        else:
            flat[full_key] = v
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set seed for Python, NumPy, and PyTorch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_dir: Optional[str] = "logs", level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout and an optional file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # stdout handler
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file handler
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(Path(log_dir) / f"{name}_{timestamp}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Timing & hardware profiling
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Context-manager timer."""
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            print(f"[Timer] {self.label}: {self.elapsed:.2f}s")


def get_gpu_memory_mb() -> Dict[str, float]:
    """Return current and peak GPU memory in MB (if CUDA available)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"current_mb": 0.0, "peak_mb": 0.0}
        current = torch.cuda.memory_allocated() / 1024 ** 2
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        return {"current_mb": round(current, 1), "peak_mb": round(peak, 1)}
    except Exception:
        return {"current_mb": 0.0, "peak_mb": 0.0}


def get_system_ram_mb() -> Dict[str, float]:
    """Return used/total system RAM in MB."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "used_mb": round(vm.used / 1024 ** 2, 1),
            "total_mb": round(vm.total / 1024 ** 2, 1),
            "percent": vm.percent,
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Results / metrics persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, output_dir: str, filename: str = "results.json") -> str:
    """Save results dict to JSON in output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return str(path)


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def count_trainable_params(model) -> Dict[str, Any]:
    """Return trainable and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
    }
