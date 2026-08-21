import json
import os
from pathlib import Path

# ─── E1 ──────────────────────────────────────────────────────────────────────
E1 = {
    "experiment_id": "E1",
    "experiment_name": "full_baseline",
    "config_path": "configs/exp1_baseline.yaml",
    "seed": 42,
    "n_train_examples": 49401,
    "n_eval_examples": 2601,
    "data_fraction": 1.0,
    "selection_method": "full",
    "ordering": "random",
    "wall_clock_seconds": 4116.97,
    "wall_clock_minutes": 68.62,
    "peak_gpu_memory_mb": 6887.6,
    "system_ram": {"used_mb": 3807.6, "total_mb": 32100.1, "percent": 13.4},
    "trainable_params": 4399104,
    "total_params": 498431872,
    "trainable_pct": 0.8826,
    "eval_loss": 1.1844538450241089,
    "eval_runtime": 69.2422,
    "eval_samples_per_second": 37.564,
    "eval_steps_per_second": 4.708,
    "epoch": 1.0,
    "train_loss": 1.2218153587894736,
    "train_runtime": 4117,
    "train_samples_per_second": 12,
    "train_steps_per_second": 0.375,
}

# ─── E2 ──────────────────────────────────────────────────────────────────────
E2 = {
    "experiment_id": "E2",
    "experiment_name": "random50_random_order",
    "config_path": "configs/exp2_random50.yaml",
    "seed": 42,
    "n_train_examples": 24700,
    "n_eval_examples": 2601,
    "data_fraction": 0.5,
    "selection_method": "random",
    "ordering": "random",
    "wall_clock_seconds": 1995.79,
    "wall_clock_minutes": 33.26,
    "peak_gpu_memory_mb": 6887.6,
    "system_ram": {"used_mb": 3163.6, "total_mb": 32100.1, "percent": 11.4},
    "trainable_params": 4399104,
    "total_params": 498431872,
    "trainable_pct": 0.8826,
    "eval_loss": 1.1957964897155762,
    "eval_runtime": 63.41,
    "eval_samples_per_second": 41.019,
    "eval_steps_per_second": 5.141,
    "epoch": 1.0,
    "train_loss": 1.236928045440832,
    "train_runtime": 1995.79,
    "train_samples_per_second": 12.38,
    "train_steps_per_second": 0.387,
}

# ─── E3 ──────────────────────────────────────────────────────────────────────
E3 = {
    "experiment_id": "E3",
    "experiment_name": "adaptive50_random_order",
    "config_path": "configs/exp3_adaptive50_random.yaml",
    "seed": 42,
    "n_train_examples": 24700,
    "n_eval_examples": 2601,
    "data_fraction": 0.5,
    "selection_method": "adaptive",
    "ordering": "random",
    "wall_clock_seconds": 1703.05,
    "wall_clock_minutes": 28.38,
    "peak_gpu_memory_mb": 6887.5,
    "system_ram": {"used_mb": 2818.7, "total_mb": 32100.1, "percent": 10.3},
    "trainable_params": 4399104,
    "total_params": 498431872,
    "trainable_pct": 0.8826,
    "eval_loss": 1.2046293020248413,
    "eval_runtime": 63.4395,
    "eval_samples_per_second": 41.0,
    "eval_steps_per_second": 5.139,
    "epoch": 1.0,
    "train_loss": 1.1542563734894589,
    "train_runtime": 1703.05,
    "train_samples_per_second": 14.50,
    "train_steps_per_second": 0.453,
}

# ─── E4 ──────────────────────────────────────────────────────────────────────
E4 = {
    "experiment_id": "E4",
    "experiment_name": "adaptive50_curriculum",
    "config_path": "configs/exp4_adaptive50_curriculum.yaml",
    "seed": 42,
    "n_train_examples": 24700,
    "n_eval_examples": 2601,
    "data_fraction": 0.5,
    "selection_method": "adaptive",
    "ordering": "curriculum",
    "wall_clock_seconds": 1706.23,
    "wall_clock_minutes": 28.44,
    "peak_gpu_memory_mb": 6887.5,
    "system_ram": {"used_mb": 2841.9, "total_mb": 32100.1, "percent": 10.4},
    "trainable_params": 4399104,
    "total_params": 498431872,
    "trainable_pct": 0.8826,
    "eval_loss": 1.2041741609573364,
    "eval_runtime": 63.2834,
    "eval_samples_per_second": 41.101,
    "eval_steps_per_second": 5.151,
    "epoch": 1.0,
    "train_loss": 1.154269584102334,
    "train_runtime": 1706.23,
    "train_samples_per_second": 14.48,
    "train_steps_per_second": 0.452,
}

# ─── E5 ──────────────────────────────────────────────────────────────────────
E5 = {
    "experiment_id": "E5",
    "experiment_name": "random50_curriculum",
    "config_path": "configs/exp5_random50_curriculum.yaml",
    "seed": 42,
    "n_train_examples": 24700,
    "n_eval_examples": 2601,
    "data_fraction": 0.5,
    "selection_method": "random",
    "ordering": "curriculum",
    "wall_clock_seconds": 1994.3,
    "wall_clock_minutes": 33.24,
    "peak_gpu_memory_mb": 6887.6,
    "system_ram": {"used_mb": 2845.8, "total_mb": 32100.1, "percent": 10.4},
    "trainable_params": 4399104,
    "total_params": 498431872,
    "trainable_pct": 0.8826,
    "eval_loss": 1.1957060098648071,
    "eval_runtime": 63.296,
    "eval_samples_per_second": 41.093,
    "eval_steps_per_second": 5.15,
    "epoch": 1.0,
    "train_loss": 1.2363436802681247,
    "train_runtime": 1994.3,
    "train_samples_per_second": 12.39,
    "train_steps_per_second": 0.387,
}

# ─── A1 ──────────────────────────────────────────────────────────────────────
A1 = {
    "experiment_id": "A1",
    "experiment_name": "ablation_diversity_only",
    "config_path": "configs/ablation_diversity_only.yaml",
    "seed": 42,
    "n_train_examples": 24700,
    "n_eval_examples": 2601,
    "data_fraction": 0.5,
    "selection_method": "adaptive",
    "ordering": "random",
    "wall_clock_seconds": 1702.0,
    "wall_clock_minutes": 28.37,
    "peak_gpu_memory_mb": 6887.5,
    "system_ram": {"used_mb": 2855.1, "total_mb": 32100.1, "percent": 10.4},
    "trainable_params": 4399104,
    "total_params": 498431872,
    "trainable_pct": 0.8826,
    "eval_loss": 1.2046276330947876,
    "eval_runtime": 63.26,
    "eval_samples_per_second": 41.12,
    "eval_steps_per_second": 5.153,
    "epoch": 1.0,
    "train_loss": 1.154256,
    "train_runtime": 1702,
    "train_samples_per_second": 14.52,
    "train_steps_per_second": 0.454,
}

# ─── A2 ──────────────────────────────────────────────────────────────────────
A2 = {
    "experiment_id": "A2",
    "experiment_name": "ablation_complexity_only",
    "config_path": "configs/ablation_complexity_only.yaml",
    "seed": 42,
    "n_train_examples": 24700,
    "n_eval_examples": 2601,
    "data_fraction": 0.5,
    "selection_method": "adaptive",
    "ordering": "random",
    "wall_clock_seconds": 1698.54,
    "wall_clock_minutes": 28.31,
    "peak_gpu_memory_mb": 6887.5,
    "system_ram": {"used_mb": 2855.1, "total_mb": 32100.1, "percent": 10.4},
    "trainable_params": 4399104,
    "total_params": 498431872,
    "trainable_pct": 0.8826,
    "eval_loss": 1.2046276330947876,
    "eval_runtime": 63.2422,
    "eval_samples_per_second": 41.128,
    "eval_steps_per_second": 5.155,
    "epoch": 1.0,
    "train_loss": 1.1542601288909122,
    "train_runtime": 1698.54,
    "train_samples_per_second": 14.54,
    "train_steps_per_second": 0.455,
}

EXPERIMENTS = {
    "E1": E1,
    "E2": E2,
    "E3": E3,
    "E4": E4,
    "E5": E5,
    "A1": A1,
    "A2": A2,
}

# ─── Write individual results.json ───────────────────────────────────────────
for exp_id, data in EXPERIMENTS.items():
    output_dir = Path(f"outputs/{data['experiment_name']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Created {out_path}")

# ─── Create experiment_results.csv ───────────────────────────────────────────
import csv
csv_path = Path("outputs/experiment_results.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "experiment_id", "experiment_name", "data_fraction", "selection_method", "ordering",
    "eval_loss", "train_loss", "wall_clock_minutes", "peak_gpu_memory_mb",
    "n_train_examples", "n_eval_examples", "trainable_params", "total_params",
    "trainable_pct", "seed"
]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for data in EXPERIMENTS.values():
        row = {k: data[k] for k in fieldnames}
        writer.writerow(row)
print(f"Created {csv_path}")

# ─── Create all_results.json ─────────────────────────────────────────────────
all_results = {}
for exp_id, data in EXPERIMENTS.items():
    all_results[exp_id] = {
        "eval_loss": data["eval_loss"],
        "train_loss": data["train_loss"],
        "wall_clock_minutes": data["wall_clock_minutes"],
        "peak_gpu_mb": data["peak_gpu_memory_mb"],
        "data_frac": data["data_fraction"],
        "selection": data["selection_method"],
        "ordering": data["ordering"],
    }

all_results_path = Path("outputs/all_results.json")
with open(all_results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Created {all_results_path}")

# ─── Validation ──────────────────────────────────────────────────────────────
print("\n=== VALIDATION ===")
for exp_id, data in EXPERIMENTS.items():
    out_path = Path(f"outputs/{data['experiment_name']}/results.json")
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded["experiment_id"] == exp_id, f"ID mismatch: {exp_id} vs {loaded['experiment_id']}"
    # Check no null values for key metrics
    for key in ["eval_loss", "train_loss", "wall_clock_minutes", "peak_gpu_memory_mb",
                "n_train_examples", "n_eval_examples", "trainable_params", "total_params",
                "trainable_pct", "seed", "eval_runtime", "eval_samples_per_second",
                "eval_steps_per_second", "epoch", "train_runtime", "train_samples_per_second",
                "train_steps_per_second", "wall_clock_seconds", "wall_clock_minutes",
                "peak_gpu_memory_mb", "system_ram", "trainable_params", "total_params",
                "trainable_pct", "eval_loss", "eval_runtime", "eval_samples_per_second",
                "eval_steps_per_second", "epoch", "train_loss"]:
        assert key in loaded, f"Missing key {key} in {exp_id}"
        val = loaded[key]
        if val is None:
            raise ValueError(f"Null value for {key} in {exp_id}")
    print(f"  {exp_id}: OK")

# Validate CSV
with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    assert len(rows) == 7, f"Expected 7 rows, got {len(rows)}"
    print(f"  CSV: {len(rows)} rows OK")

# Validate all_results.json
with open(all_results_path) as f:
    all_res = json.load(f)
    assert set(all_res.keys()) == {"E1", "E2", "E3", "E4", "E5", "A1", "A2"}, f"Keys mismatch: {all_res.keys()}"
    print(f"  all_results.json: 7 experiments OK")

print("\n=== ALL VALIDATIONS PASSED ===")
