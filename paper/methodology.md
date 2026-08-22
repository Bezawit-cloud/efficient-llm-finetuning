# Methodology and Experimental Setup

This document specifies the methodology **exactly as implemented** in this repository. It is written from the source code (`src/`, `configs/`) rather than from design intentions; where the implementation admits ambiguity or where repository evidence is incomplete, this is stated explicitly. This document contains method and setup only; results, discussion, and conclusions are presented separately.

---

## 1. Problem Formulation

Supervised instruction tuning fine-tunes a pretrained generative language model on a corpus of (instruction, input, response) triples. Training cost scales linearly in the number of training examples (per epoch), motivating the question studied here: *can a subset of training examples retain most of the benefit of the full corpus while reducing training cost?*

Let $\mathcal{D} = \{x_1, \dots, x_N\}$ denote the full instruction-tuning corpus ($N$ = 49,401 training examples after the split described in §2). We study two orthogonal interventions:

- **Selection** — choosing a subset $S \subseteq \mathcal{D}$ with $|S| = f \cdot N$, where the fraction $f$ is fixed at $f = 0.5$ for all subset experiments. Selection strategies compared: the full corpus ($f=1$), uniform random subsampling, and importance-score-based ("adaptive") top-$k$ selection.
- **Ordering** — given a selected subset, deciding the sequence in which examples are presented within an epoch: uniformly shuffled ("random") versus sorted by increasing estimated difficulty ("curriculum").

Selection and ordering are independent factors. The experimental design (§8) crosses them so that each factor's effect can be isolated at a fixed compute budget.

One auxiliary question is addressed by ablations (§8): whether the combined importance score owes its effect to any single component — i.e., how each scoring component contributes individually.

---

## 2. Dataset

| Property | Value |
|---|---|
| Dataset | `tatsu-lab/alpaca` (Hugging Face Datasets) |
| Total examples | 52,002 |
| Split | 95% train / 5% evaluation, single random split |
| Train examples | 49,401 |
| Evaluation examples | 2,601 |
| Split seed | 42 |
| Deduplication / cleaning | None applied |

The dataset is loaded via `datasets.load_dataset("tatsu-lab/alpaca", split="train")` (`src/data_utils.py`). The train/evaluation partition is produced once with `train_test_split(test_size=0.05, seed=42)`; all seven experiments share this identical partition. Each example is rendered into a fixed prompt template containing the instruction, an optional input block (present when the example provides one), and the target response.

No deduplication, filtering, or text cleaning is applied beyond the template rendering above; Alpaca is used in its distributed form. The held-out evaluation set is excluded from importance scoring and from subset selection in every experiment: scores are computed over the training split only, and both centroid statistics and min–max normalization ranges (§3) are derived exclusively from that split.

**Reproducibility gap:** the Hugging Face dataset revision (commit hash) is not pinned in the configuration; the dataset is referenced by name only.

---

## 3. Importance Scoring

Every training example receives three sub-scores, each normalized to $[0, 1]$ over the training split, combined into a scalar importance score. All components are lightweight heuristics computed without any language-model judge.

### 3.1 Semantic diversity $D(x)$

Each example's instruction (with its input, if present) is embedded with `all-MiniLM-L6-v2` (sentence-transformers), producing unit-normalized embedding vectors. Let $\mu = \frac{1}{N}\sum_i e_i$ be the corpus centroid. With L2-normalized embeddings, cosine similarity equals the inner product:

$$D(x_i) = 1 - e_i^\top \mu,$$

followed by min–max normalization across the training split. High scores therefore indicate examples far from the distributional center — under-represented directions in embedding space.

*Implementation note:* if sentence-transformers is unavailable, the code falls back to TF-IDF representations (5,000 features, sublinear TF) with the same centroid-distance definition. In the executed experiments the embedding path was used; the fallback was not exercised.

### 3.2 Instruction complexity $C(x)$

Two signals are computed from the instruction concatenated with its input field:

- character length $\ell(x)$;
- vocabulary richness $v(x)$ = number of unique whitespace-delimited lowercase tokens divided by total token count.

Each signal is min–max normalized independently; their arithmetic mean is taken; the result is min–max normalized again:

$$C(x) = \mathrm{minmax}\!\left(\tfrac{1}{2}\big[\mathrm{minmax}(\ell)(x) + \mathrm{minmax}(v)(x)\big]\right).$$

This is deliberately a shallow proxy for task difficulty: it responds to surface length and lexical variety, not to semantic depth.

### 3.3 Response-length component $R(x)$

Let $r(x)$ be the character length of the gold response. The score applies a smooth penalty to extremes on a log scale:

$$R(x) = \mathrm{clip}_{[0,1]}\!\left(1 - \frac{\left|\mathrm{minmax}(\log(1+r))(x) - 0.6\right|}{0.6}\right).$$

The design intent is a bell-shaped reward peaking in the mid-range of observed response lengths (approximately 200–800 characters in Alpaca): very short responses are treated as low-information and extremely long responses as potential padding. The 0.6 target was chosen a priori from the dataset's approximate mid-range and was not tuned.

### 3.4 Combined importance score

$$\mathrm{Importance}(x) = \alpha D(x) + \beta C(x) + \gamma R(x), \qquad \alpha = 0.333,\ \beta = 0.333,\ \gamma = 0.334.$$

The weights were **fixed a priori** (uniform thirds) and were **not tuned against any evaluation result**; no weight sweep was performed. Scores are computed once over the entire training split and cached to `data/scored_alpaca.json`; all experiments read the same cached scores.

---

## 4. Selection

Three selection mechanisms are implemented (`src/select_and_order.py`):

1. **Full-data selection** — the entire training split is used ($f = 1.0$).
2. **Uniform random selection** — $k = \lfloor f \cdot N \rfloor$ examples drawn uniformly without replacement using Python's `random.Random(seed)` with the experiment seed ($k$ = 24,700 at $f=0.5$).
3. **Adaptive (importance-based) selection** — examples are ranked by descending combined importance score and the top $k$ retained. Ties are broken deterministically by Python's stable sort order.

Random draws are fully determined by the seed: the same seed reproduces the identical subset across runs and across experiments sharing a configuration. Ordering uses a derived seed offset (seed + 1) so that shuffling is seeded yet decorrelated from the selection draw.

---

## 5. Ordering / Curriculum

Given the selected subset $S$, two orderings are implemented:

- **Random ordering** — a seeded uniform shuffle of $S$.
- **Easy-to-hard curriculum** — $S$ is sorted by **ascending complexity sub-score** $C(x)$, presenting estimated-easy examples first and estimated-hard examples last within each epoch.

Both orderings are **static**: the presentation order is fixed before training begins and is identical in every epoch. There is no dynamic difficulty estimation, no re-scoring during training, and no pacing schedule (e.g., no window resizing or example dropping). The curriculum signal is the same heuristic complexity measure used as one component of the selection score (§3.2); this non-orthogonality between selection and ordering signals is acknowledged as a design property rather than hidden.

---

## 6. Model and Fine-Tuning

| Component | Configuration |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Total parameters | 498,431,872 |
| Adaptation | LoRA (PEFT), causal-LM task type |
| LoRA rank / alpha | 8 / 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (all seven projections) |
| LoRA bias | none |
| Trainable parameters | 4,399,104 (0.8826%) |
| Optimizer | AdamW (`adamw_torch`) |
| Learning rate | 2×10⁻⁴ |
| Schedule | Cosine decay with 3% warmup (ratio of total steps) |
| Weight decay | 0.001 |
| Epochs | 1 |
| Micro-batch size | 8 (train and eval) |
| Gradient accumulation | 4 |
| Effective batch size | 32 |
| Maximum sequence length | 384 tokens (truncation) |
| Padding | Dynamic, per-batch, via `DataCollatorForSeq2Seq`; padded label positions masked with −100 |
| Precision | FP16 (mixed precision) |
| Gradient checkpointing | Enabled |
| Seed | 42 |

Training uses Hugging Face `Trainer`. The base model weights are frozen; only LoRA adapters receive gradients. Model weights are loaded with dtype `"auto"` (FP16 on the target GPU). Tokenizer padding uses the EOS token.

---

## 7. Loss Formulation (Important for Interpretation)

Training and evaluation use **full-sequence language-modeling loss**: token labels equal the complete input sequence (instruction, optional input block, response, EOS). Instruction tokens are **not masked out**; the model receives cross-entropy supervision on every non-padding token, including the prompt portion of each example.

Consequence for interpretation: `eval_loss` is the mean cross-entropy of the held-out set under this full-sequence objective. It measures how well the adapted model predicts held-out Alpaca sequences — including their instructions — under the identical objective used in training. It is **not** a direct measure of instruction-following quality, response fluency, factual accuracy, or generation adequacy; no generation was sampled and no generation-based metric (ROUGE/BLEU/win-rate/human judgment) was computed anywhere in this study. Because the loss formulation is identical across all experimental conditions, relative comparisons between conditions remain internally meaningful; absolute values should not be compared to studies using completion-only loss masking.

---

## 8. Experimental Design

### 8.1 Main experiments (selection × ordering)

Seven runs share the model, LoRA configuration, hyperparameters, data split, evaluation set, and seed (§6, §2). Only the selection strategy, subset fraction, and ordering differ.

| Experiment | Selection strategy | Fraction | Ordering | Role |
|---|---|---|---|---|
| E1 | Full data (no selection) | 1.00 | Random | Upper reference for quality; anchor for efficiency claims |
| E2 | Uniform random | 0.50 | Random | Control: effect of halving data volume alone |
| E3 | Adaptive (top-50% by importance) | 0.50 | Random | Effect of scored selection vs random at equal budget |
| E4 | Adaptive (top-50%) | 0.50 | Easy→hard curriculum | Proposed full method: selection + ordering |
| E5 | Uniform random | 0.50 | Easy→hard curriculum | Isolates ordering effect without selection |

Planned contrasts: **E2 vs E3** isolates selection; **E3 vs E4** isolates ordering on adaptively selected data; **E2 vs E5** isolates ordering on randomly selected data; **E1 vs E2** quantifies the cost of halving the corpus.

### 8.2 Scoring-component ablations

| Experiment | Score weights | Selection | Fraction | Ordering | Purpose |
|---|---|---|---|---|---|
| A1 | α=1, β=0, γ=0 (diversity only) | Adaptive | 0.50 | Random | Diversity component alone |
| A2 | β=1, α=0, γ=0 (complexity only) | Adaptive | 0.50 | Random | Complexity component alone |

A1 and A2 are **ablations of E3's adaptive selection rule**, not additional cells of the main selection × ordering matrix. They share E3's fraction and ordering so that each can be compared directly against E3: agreement with E3 would indicate the dropped components contribute nothing beyond the retained one. The response-length component (γ alone) was not ablated separately.

---

## 9. Evaluation

All experiments are evaluated identically:

- Held-out evaluation set: **2,601 Alpaca examples**, disjoint from the training split (§2), never seen by scoring or selection.
- Metric: **eval_loss** — cross-entropy under the full-sequence objective (§7), computed once after training completes.
- Not evaluated: generation quality metrics (ROUGE, BLEU, BERTScore), human preference or win-rate judgments, downstream task benchmarks, factuality, safety.

Eval-loss comparisons across conditions are like-for-like (identical data, objective, and decoding-free protocol); the metric's limitations as a proxy for instruction-following quality are discussed separately.

---

## 10. Efficiency Measurement

All seven canonical runs executed sequentially on the same cloud instance: a single **NVIDIA Tesla T4** (15,360 MiB VRAM), Kaggle environment, FP16 compute. Measured quantities per run:

- **Wall-clock training time**: `time.perf_counter()` bracketing `Trainer.train()` only.
  - *Included:* forward/backward passes, gradient accumulation, optimizer steps, gradient-checkpoint recomputation, periodic logging.
  - *Excluded:* model/tokenizer download and loading, dataset download, tokenization, importance scoring, subset construction, checkpoint serialization, and the post-training evaluation pass.
- **Peak GPU memory**: `torch.cuda.max_memory_allocated()`, reset immediately before training. Because the counter is read after evaluation completes, the reported peak spans training **and** the final evaluation pass; in practice training dominates this value.
- **System RAM** (via psutil) recorded as auxiliary telemetry.

Timing comparisons are between runs on the same hardware and session type; absolute times should not be extrapolated to other devices.

---

## 11. Reproducibility

Elements supporting reproduction:

- **Seeding:** global seed 42 applied to Python `random`, NumPy, PyTorch (CPU and all CUDA devices), with deterministic cuDNN flags set; ordering uses seed + 1. Per-experiment seeds are identical (42).
- **Configuration:** all hyperparameters and per-experiment overrides live in version-controlled YAML files (`configs/base_config.yaml`, `configs/exp*.yaml`, `configs/ablation_*.yaml`) merged at launch time.
- **Entry point:** `src/train_baseline.py --config configs/<experiment>.yaml` executes scoring, selection, ordering, training, evaluation, and artifact writing end-to-end.
- **Executed record:** `notebooks/notebook623351e51a.ipynb` preserves the complete executed Kaggle session (logs and saved JSON outputs) for all seven runs. It contains Kaggle-specific absolute paths and is an execution record, not a portable runner; `notebooks/cloud_run.ipynb` is the portable orchestration notebook.
- **Result artifacts:** `outputs/<experiment>/results.json` (full precision), aggregated as `outputs/experiment_results.csv` and `outputs/all_results.json`.
- **Figures:** `scripts/generate_figures.py` regenerates all five figures (PNG + PDF) solely from the result artifacts; no metric values are hard-coded.
- **Provenance notes (README):** the canonical records correspond to the executed suite session; two reconciliation corrections (suite-run E1 adopted as canonical; A1 loss values recovered from the notebook after a reconstruction collision with A2) and one field-level limitation (A1 `system_ram` not independently verifiable) are documented. A standalone same-configuration E1 repeat exists and is documented as an observed same-seed repeat difference (~2×10⁻⁵ eval loss), not as a variance estimate.

Known reproducibility gaps (documented, unresolved):

1. Hugging Face dataset and model revisions are not pinned (referenced by name only).
2. Exact package versions are not locked; `requirements.txt` states minimum versions only.
3. Run-to-run variation on identical configuration was not systematically characterized (a single repeat pair suggests ~10⁻⁵-scale eval-loss differences).

These gaps qualify perfect bit-level reproduction but do not affect the internal validity of the recorded comparison, which was executed under a single controlled session.

---

*Scope statement: everything in this document describes Qwen2.5-0.5B-Instruct fine-tuned with LoRA (r=8, α=16) on Alpaca for one epoch under the configuration above. No claim in the accompanying paper extends beyond this setting.*
