# Adaptive Data Selection & Curriculum Learning for Compute-Efficient LLM Fine-Tuning

> **Headline result:** In this setup, a random 50% subset of Alpaca reached eval loss within +0.0113 of the full-data baseline while reducing training wall-clock time by ~49.0%. Adaptive heuristic selection and curriculum ordering provided no measurable benefit over random selection at this scale. All findings are from single-seed runs (seed 42) on Qwen2.5-0.5B-Instruct + LoRA + Alpaca (1 epoch) — see [Experimental Findings](#experimental-findings).

A research project investigating whether smart data selection + curriculum ordering can match full-dataset fine-tuning quality at half the compute cost.

---

## Research Question

*Does combining representativeness-based data selection with difficulty-ordered curriculum learning recover full-dataset instruction-tuning performance (measured by held-out eval loss) using only 50% of the training data — and does ordering add benefit beyond selection alone?* *(Answered for this setup: random 50% came within +0.011 eval loss of the full baseline; see Experimental Findings.)*

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
- **GPU:** $\ge$ 8 GB VRAM recommended (all experiments were run on a single NVIDIA T4)
- **Precision:** FP16 or BF16 (`torch_dtype="auto"`)
- **Measured peak VRAM:** ~6.9 GB across all seven runs
- **Measured suite runtime (1× T4):** E1 ≈ 65 min; each 50% run ≈ 28–33 min; full suite (7 runs) ≈ 4.1 hours

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
├── scripts/
│   └── generate_figures.py # Generates all figures from outputs/**/results.json
├── data/
│   └── scored_alpaca.json  # Cached importance scores (generated on first run)
├── outputs/                # results.json per experiment + experiment_results.csv + all_results.json
├── figures/                # Generated plots (PNG + PDF)
├── notebooks/              # cloud_run.ipynb (orchestration) + executed Kaggle notebook
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
| Epochs | 1 |
| Batch size | 8 (+ 4 grad accum steps = effective 32) |
| Max seq length | 384 |
| Seed | 42 |
| Dataset | tatsu-lab/alpaca (52k examples; 49,401 train / 2,601 held-out eval) |

---

## Results

All seven experiments completed on a single NVIDIA T4 (Kaggle), seed 42.
Values below are reproduced exactly from the validated artifacts in `outputs/**/results.json`.

| Exp | Selection | Ordering | Fraction | n_train | Eval Loss | Train Loss | Time (min) | Peak GPU MB | Seed |
|-----|-----------|----------|---------:|--------:|----------:|-----------:|-----------:|------------:|-----:|
| E1 | full | random | 1.0 | 49,401 | 1.184476 | 1.221839 | 65.22 | 6887.6 | 42 |
| E2 | random | random | 0.5 | 24,700 | 1.195796 | 1.236928 | 33.26 | 6887.6 | 42 |
| E3 | adaptive | random | 0.5 | 24,700 | 1.204629 | 1.154256 | 28.38 | 6887.5 | 42 |
| E4 | adaptive | curriculum | 0.5 | 24,700 | 1.204174 | 1.154270 | 28.44 | 6887.5 | 42 |
| E5 | random | curriculum | 0.5 | 24,700 | 1.195706 | 1.236344 | 33.24 | 6887.6 | 42 |
| A1 | adaptive (α=1, diversity-only) | random | 0.5 | 24,700 | 1.204624 | 1.154256 | 28.37 | 6887.5 | 42 |
| A2 | adaptive (β=1, complexity-only) | random | 0.5 | 24,700 | 1.204628 | 1.154260 | 28.31 | 6887.5 | 42 |

Eval loss is reported at 6 decimal places here for presentation; full-precision values are stored in the result artifacts (`outputs/experiment_results.csv`, `outputs/all_results.json`, and per-experiment `results.json`). Eval loss is a held-out language-modeling metric — it is not a direct measure of generation quality (no ROUGE, win-rate, or human-preference evaluation was performed).

### Experimental Findings

These observations are **descriptive**: each configuration was trained once with seed 42, so differences are not significance-tested and should not be generalized beyond this exact setup (Qwen2.5-0.5B-Instruct + LoRA r=8 + Alpaca + 1 epoch + the hyperparameters above).

- Random 50% selection retained performance close to the full-data baseline (eval-loss increase of +0.0113 relative to E1) while reducing training wall-clock time by approximately 49.0%.
- In this experimental setup, adaptive 50% selection did not outperform random 50% selection (E3 vs E2: +0.0088 eval-loss difference).
- Curriculum ordering produced no measurable improvement for either selection strategy (E3→E4: −0.0005; E2→E5: −0.0001).
- The diversity-only (A1) and complexity-only (A2) ablations were numerically indistinguishable from the combined adaptive configuration (differences ≈ 0.000006).
- These findings are specific to the tested model, dataset, scoring function, and scale; they do not establish that random selection is universally preferable or that heuristic selection fails generally.

---

## Artifact Provenance & Reproducibility

**Result artifacts.** The seven canonical `outputs/**/results.json` files correspond to the executed experiment suite recorded in `notebooks/notebook623351e51a.ipynb` (all seven runs executed sequentially on the same Kaggle T4 session with seed 42). Because the original Kaggle working directory was no longer directly available, the JSON records were reconstructed from the verified notebook/log outputs; they were subsequently cross-validated for exact numerical agreement against:

- `outputs/experiment_results.csv`
- `outputs/all_results.json`
- the executed Kaggle notebook

Two provenance corrections were applied during reconciliation with the executed notebook: (1) the canonical E1 record uses the suite-run baseline rather than an earlier standalone E1 validation run of the identical configuration; (2) the A1 record's loss values, which had collided with A2's during initial reconstruction, were corrected from the authoritative notebook output. A standalone same-configuration E1 repeat also exists (eval_loss 1.1844538450241089, 68.62 min); it is retained here as documentation of observed run-to-run variation (~0.00002 eval-loss difference between same-seed repeats) but is not part of the primary seven-experiment table. No training was rerun during artifact correction.

`create_results.py` records how the JSONs were reconstructed and holds the canonical values.

**Figures.** [`scripts/generate_figures.py`](scripts/generate_figures.py) generates all figures dynamically from the validated result artifacts — no metric values are hard-coded. It produces PNG + PDF versions of:

- `figures/fig1_eval_loss_comparison`
- `figures/fig2_eval_loss_vs_time`
- `figures/fig3_data_fraction_vs_eval_loss`
- `figures/fig4_curriculum_comparison`
- `figures/fig5_adaptive_selection_ablation`

Run it from the repository root with any Python environment that has matplotlib installed (it reads only `outputs/**/results.json` and writes only to `figures/`):

```bash
python scripts/generate_figures.py
```

**Notebook limitation.** The executed Kaggle notebook (`notebooks/notebook623351e51a.ipynb`) contains Kaggle-specific absolute `/kaggle/working/...` paths in its results-compilation step and therefore is not guaranteed to run outside Kaggle without path adaptation. [`notebooks/cloud_run.ipynb`](notebooks/cloud_run.ipynb) is the portable orchestration notebook for re-running the experiment suite on Kaggle.

---

## Citation / Related Work

- **DEITA** (Liu et al., 2024) — complexity/diversity-based instruction data selection at 7B–13B scale. This project adapts the scoring idea to sub-1B PEFT scale with lightweight heuristic scorers and additionally evaluates the effect of curriculum ordering on top of selection.
