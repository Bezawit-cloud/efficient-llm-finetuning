# Adaptive Data Selection & Curriculum Learning for Compute-Efficient LLM Fine-Tuning

> **Headline result (TBD after experiments):** _X%_ of full-dataset performance using 50% of the data and _Y%_ less training time than random selection alone.

A research project investigating whether smart data selection + curriculum ordering can match full-dataset fine-tuning quality at half the compute cost.

---

## Research Question

*Does combining representativeness-based data selection with difficulty-ordered curriculum learning recover ≥X% of full-dataset instruction-following performance using only 50% of the training data — and does ordering add benefit beyond selection alone?*

---

## Method Overview

Each training example is scored by an **importance score**:

```
Importance = α·Diversity + β·Complexity + γ·ResponseLength
           = 0.333·D + 0.333·C + 0.334·R
```

- **Diversity** — embedding distance from the dataset centroid (far = more unique)
- **Complexity** — instruction length + vocabulary richness (lightweight heuristic, no LLM needed)
- **ResponseLength** — bell-shaped reward favouring responses in the ~200–800 char range

The top-50% by importance score are selected (adaptive selection), then optionally sorted **easy→hard** by complexity (curriculum ordering).

This is a DEITA-inspired approach adapted for sub-1B PEFT scale with lightweight heuristic scorers instead of 7B–13B LLM-based judges.

---

## Experiments (2×2 Design)

| Exp | Selection | Ordering | Description |
|-----|-----------|----------|-------------|
| **E1** | 100% (full) | random | Baseline — full Alpaca dataset |
| **E2** | Random 50% | random | Random subset, no ordering |
| **E3** | Adaptive 50% | random | Smart selection, no ordering |
| **E4** | Adaptive 50% | easy→hard | **Full method** (selection + curriculum) |
| **E5** | Random 50% | easy→hard | Curriculum on random subset (isolation test) |
| **Ablation A** | Adaptive 50% | random | Diversity score only |
| **Ablation B** | Adaptive 50% | random | Complexity score only |

**Isolation logic:**
- E2 vs E3 → effect of *selection*
- E3 vs E4 → effect of *curriculum ordering*
- E2 vs E5 → curriculum on unselected data

---

## Setup

### Cloud GPU Environment Setup (Recommended)

1. **Clone repository:**
   ```bash
   git clone <REPO_URL>
   cd efficient-llm-finetuning
   ```

2. **Install Dependencies (CUDA-enabled):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU with Smoke Test:**
   ```bash
   python src/gpu_smoke_test.py
   ```
   *This verifies CUDA detection, loads `Qwen/Qwen2.5-0.5B-Instruct`, attaches LoRA, runs 1 training step, and logs peak VRAM to `outputs/smoke_test/smoke_test_results.json`.*

### Interactive Cloud Notebook
Alternatively, open and run [`notebooks/cloud_run.ipynb`](notebooks/cloud_run.ipynb) directly in Google Colab (Free T4 or Pro A100), Kaggle, or RunPod.

### Hardware Requirements
- **GPU:** $\ge$ 6–16 GB VRAM (e.g., NVIDIA T4, RTX 3090, RTX 4090, A10G, A100, L4)
- **Precision:** FP16 or BF16 (`torch_dtype="auto"`)
- **Measured VRAM footprint:** ~2.5 GB to 3.5 GB peak
- **Estimated Full Suite Runtime:** ~45–65 minutes (all 5 primary runs + 2 ablations + scoring)

---

## Run Commands

### Sanity check (dry run — 100 examples, 1 epoch, ~2 min)
```bash
python src/train_baseline.py --config configs/exp1_baseline.yaml --dry_run
```

### Full experiments
```bash
# E1 — Baseline (100% data, random order)
python src/train_baseline.py --config configs/exp1_baseline.yaml

# E2 — Random 50%, random order
python src/train_baseline.py --config configs/exp2_random50.yaml

# E3 — Adaptive 50%, random order
python src/train_baseline.py --config configs/exp3_adaptive50_random.yaml

# E4 — Adaptive 50% + curriculum (FULL METHOD)
python src/train_baseline.py --config configs/exp4_adaptive50_curriculum.yaml

# E5 — Random 50% + curriculum (isolation test)
python src/train_baseline.py --config configs/exp5_random50_curriculum.yaml

# Ablation: diversity score only
python src/train_baseline.py --config configs/ablation_diversity_only.yaml

# Ablation: complexity score only
python src/train_baseline.py --config configs/ablation_complexity_only.yaml
```

### Override seed
```bash
python src/train_baseline.py --config configs/exp4_adaptive50_curriculum.yaml --seed 123
```

---

## Repository Structure

```
efficient-llm-finetuning/
├── configs/
│   ├── base_config.yaml              # Shared hyperparameters for all experiments
│   ├── exp1_baseline.yaml            # E1: 100% data, random order
│   ├── exp2_random50.yaml            # E2: random 50%
│   ├── exp3_adaptive50_random.yaml   # E3: adaptive 50%, random order
│   ├── exp4_adaptive50_curriculum.yaml  # E4: adaptive 50% + curriculum
│   ├── exp5_random50_curriculum.yaml    # E5: random 50% + curriculum
│   ├── ablation_diversity_only.yaml  # Ablation: diversity only
│   └── ablation_complexity_only.yaml # Ablation: complexity only
├── src/
│   ├── train_baseline.py   # Main training entry-point (all experiments)
│   ├── scoring.py          # Importance scoring (diversity, complexity, response length)
│   ├── select_and_order.py # Data selection + curriculum ordering
│   ├── data_utils.py       # Alpaca dataset loading + tokenization
│   └── utils.py            # Config loading, logging, timing, results saving
├── data/
│   └── scored_alpaca.json  # Cached importance scores (generated on first run)
├── outputs/                # Checkpoints + results.json per experiment
├── figures/                # Generated plots
├── paper/                  # Paper draft
└── requirements.txt
```

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Target modules | q, k, v, o, gate, up, down proj |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Batch size | 4 (+ 4 grad accum steps = effective 16) |
| Max seq length | 512 |
| Seed | 42 |
| Dataset | tatsu-lab/alpaca (52k examples) |

---

## Results (populated after experiments)

| Exp | Data % | Ordering | Eval Loss | Train Time (min) | Peak GPU Mem (MB) |
|-----|--------|----------|-----------|-----------------|-------------------|
| E1  | 100%   | random   | TBD       | TBD             | TBD               |
| E2  | 50%    | random   | TBD       | TBD             | TBD               |
| E3  | 50%    | random   | TBD       | TBD             | TBD               |
| E4  | 50%    | curriculum | TBD     | TBD             | TBD               |
| E5  | 50%    | curriculum | TBD     | TBD             | TBD               |

---

## Citation / Related Work

- **DEITA** (Liu et al., 2024) — complexity/diversity-based instruction data selection at 7B–13B scale. This project adapts the scoring idea to sub-1B PEFT scale with lightweight heuristic scorers and additionally evaluates the effect of curriculum ordering on top of selection.
