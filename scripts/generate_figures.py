#!/usr/bin/env python3
"""
generate_figures.py — Step 7 figure generation for efficient-llm-finetuning.

Reads ONLY the validated experiment artifacts under outputs/**/results.json,
generates five publication-ready figures (PNG + PDF) into figures/, and
validates its own output.

Usage (from repository root):
    python scripts/generate_figures.py

Notes:
    - No metric values are hard-coded; everything is loaded from the JSONs.
    - This script never writes to outputs/, configs/, src/, or data/.
    - All experiments are single runs with seed 42; figures state this and
      make no statistical-significance claims.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
SEED_NOTE = "Single run per configuration (seed = 42); differences are descriptive, not significance-tested."

EXP_ORDER = ["E1", "E2", "E3", "E4", "E5", "A1", "A2"]

# Visual identity, consistent across all figures
COLORS = {
    "full": "#2ca02c",        # green  – full-data baseline
    "random": "#1f77b4",      # blue   – random selection
    "adaptive": "#ff7f0e",    # orange – adaptive selection
}
MARKERS = {
    "E1": "o",
    "E2": "s",
    "E5": "s",
    "E3": "^",
    "E4": "^",
    "A1": "D",
    "A2": "D",
}
GROUP_STYLE = dict(
    full=("o", "#2ca02c"),
    random=("s", "#1f77b4"),
    adaptive=("^", "#ff7f0e"),
)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────
def load_results() -> dict[str, dict]:
    """Load every outputs/**/results.json keyed by experiment_id."""
    pattern = str(REPO_ROOT / "outputs" / "**" / "results.json")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        sys.exit(f"ERROR: no results.json found under {REPO_ROOT / 'outputs'}")

    exp: dict[str, dict] = {}
    for f in files:
        with open(f) as fp:
            rec = json.load(fp)
        eid = rec["experiment_id"]
        if eid in exp:
            sys.exit(f"ERROR: duplicate experiment_id {eid} ({f})")
        rec["_source_file"] = f
        exp[eid] = rec

    missing = [e for e in EXP_ORDER if e not in exp]
    if missing:
        sys.exit(f"ERROR: missing experiments: {missing}")
    return {e: exp[e] for e in EXP_ORDER}


def color_of(eid: str) -> str:
    return COLORS[exp[eid]["selection_method"]]


def marker_of(eid: str) -> str:
    return MARKERS[eid]


def save(fig, name: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(str(p))
    plt.close(fig)
    return paths


def add_seed_note(fig):
    fig.text(0.99, 0.01, SEED_NOTE, ha="right", va="bottom",
             fontsize=7.5, color="0.35", style="italic")


def short_desc(eid: str) -> str:
    r = exp[eid]
    sel = {"full": "Full data", "random": "Random 50%", "adaptive": "Adaptive 50%"}[
        r["selection_method"]
    ]
    order = "curriculum" if r["ordering"] == "curriculum" else "random order"
    extra = ""
    if eid == "A1":
        extra = " [diversity-only]"
    elif eid == "A2":
        extra = " [complexity-only]"
    return f"{eid}: {sel}, {order}{extra}"


# ────────────────────────────────────────────────────────────────────────────
# Figures
# ────────────────────────────────────────────────────────────────────────────
def fig1_eval_loss_comparison():
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ids = EXP_ORDER
    vals = [exp[e]["eval_loss"] for e in ids]

    ax.barh(range(len(ids)), vals,
            color=[color_of(e) for e in ids],
            edgecolor="black", linewidth=0.6, height=0.62)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([short_desc(e) for e in ids], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Eval loss on held-out Alpaca split (lower is better)")
    ax.set_title("Instruction-tuning eval loss across selection / ordering strategies")

    for i, v in enumerate(vals):
        ax.text(v + 0.0008, i, f"{v:.6f}", va="center", fontsize=8.5)

    base = exp["E1"]["eval_loss"]
    ax.axvline(base, color="red", ls="--", lw=1, alpha=0.6,
               label=f"E1 full-data baseline ({base:.6f})")
    ax.set_xlim(min(vals) - 0.006, max(vals) + 0.012)
    ax.legend(loc="lower right", fontsize=9)
    add_seed_note(fig)
    return save(fig, "fig1_eval_loss_comparison")


def fig2_eval_loss_vs_time():
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for e in EXP_ORDER:
        x = exp[e]["wall_clock_minutes"]
        y = exp[e]["eval_loss"]
        m, c = GROUP_STYLE[exp[e]["selection_method"]]
        ax.scatter(x, y, s=110, marker=m, color=c,
                   edgecolor="black", linewidth=0.6, zorder=5)
        dy = -0.0016 if e in ("A1", "E3") else 0.0009
        ax.annotate(e, (x, y), xytext=(6, 6 if dy > 0 else -11),
                    textcoords="offset points", fontsize=9, fontweight="bold")

    ax.set_xlabel("Wall-clock training time (minutes)")
    ax.set_ylabel("Eval loss (lower is better)")
    ax.set_title("Compute efficiency: eval loss vs training time")
    ax.grid(alpha=0.25)

    handles = [
        Line2D([], [], marker="o", ls="", color="#2ca02c", markeredgecolor="black",
               label="Full data baseline"),
        Line2D([], [], marker="s", ls="", color="#1f77b4", markeredgecolor="black",
               label="Random 50%"),
        Line2D([], [], marker="^", ls="", color="#ff7f0e", markeredgecolor="black",
               label="Adaptive 50%"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    add_seed_note(fig)
    return save(fig, "fig2_eval_loss_vs_time")


def fig3_data_fraction_vs_eval_loss():
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for e in EXP_ORDER:
        x = exp[e]["data_fraction"]
        y = exp[e]["eval_loss"]
        m, c = GROUP_STYLE[exp[e]["selection_method"]]
        jitter = 0.0
        # A1/A2 overlap E3 exactly at fraction 0.5 — nudge markers horizontally
        if e == "A1":
            jitter = -0.018
        elif e == "A2":
            jitter = 0.018
        ax.scatter(x + jitter, y, s=110, marker=m, color=c,
                   edgecolor="black", linewidth=0.6, zorder=5)
        ax.annotate(e, (x + jitter, y), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, fontweight="bold")

    ax.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.7)
    ax.text(1.0, ax.get_ylim()[0], " ", fontsize=1)  # keep xlim stable
    ax.set_xlim(-0.08, 1.12)
    ax.set_xticks([0.5, 1.0])
    ax.set_xticklabels(["50%\n(24,700 examples)", "100%\n(49,401 examples)"])
    ax.set_xlabel("Fraction of training data used")
    ax.set_ylabel("Eval loss (lower is better)")
    ax.set_title("Data efficiency: subset size vs eval loss")
    ax.grid(alpha=0.25, axis="y")

    handles = [
        Line2D([], [], marker="o", ls="", color="#2ca02c", markeredgecolor="black",
               label="E1: full dataset"),
        Line2D([], [], marker="s", ls="", color="#1f77b4", markeredgecolor="black",
               label="Random 50% (E2, E5)"),
        Line2D([], [], marker="^", ls="", color="#ff7f0e", markeredgecolor="black",
               label="Adaptive 50% (E3, E4)"),
        Line2D([], [], marker="D", ls="", color="#ff7f0e", alpha=0.55,
               markeredgecolor="black", label="Scoring ablations (A1, A2)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    add_seed_note(fig)
    return save(fig, "fig3_data_fraction_vs_eval_loss")


def fig4_curriculum_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
    panels = [(["E3", "E4"], "Adaptive 50%:\nrandom vs curriculum"),
              (["E2", "E5"], "Random 50%:\nrandom vs curriculum")]

    all_vals = []
    for ids in (["E3", "E4"], ["E2", "E5"]):
        all_vals += [exp[e]["eval_loss"] for e in ids]

    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 3 + 0.002

    for ax, ids, title in ((ax, ids, t) for ax, (ids, t) in zip(axes, panels)):
        vals = [exp[e]["eval_loss"] for e in ids]
        ax.bar(range(len(ids)), vals,
               color=[color_of(e) for e in ids],
               edgecolor="black", linewidth=0.6, width=0.55)
        ax.set_xticks(range(len(ids)))
        labels = []
        for e in ids:
            sel = "Adaptive" if exp[e]["selection_method"] == "adaptive" else "Random"
            order = "+ curriculum" if exp[e]["ordering"] == "curriculum" else "+ random order"
            labels.append(f"{e}\n{sel}\n{order}")
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(lo - pad, hi + pad)
        for i, v in enumerate(vals):
            ax.text(i, v + pad * 0.06, f"{v:.6f}", ha="center", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.2, axis="y")

    axes[0].set_ylabel("Eval loss (lower is better)")

    d1 = exp["E4"]["eval_loss"] - exp["E3"]["eval_loss"]
    d2 = exp["E5"]["eval_loss"] - exp["E2"]["eval_loss"]
    fig.suptitle(
        f"Curriculum ordering effect on eval loss   "
        f"(Δ E4−E3 = {d1:+.6f};  Δ E5−E2 = {d2:+.6f})",
        fontsize=11.5, fontweight="bold")
    add_seed_note(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    return save(fig, "fig4_curriculum_comparison")


def fig5_adaptive_selection_ablation():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ids = ["A1", "A2", "E3"]
    names = {
        "A1": "A1\nDiversity-only\nscoring",
        "A2": "A2\nComplexity-only\nscoring",
        "E3": "E3\nCombined score\n(D+C+R)",
    }
    vals = [exp[e]["eval_loss"] for e in ids]
    times = [exp[e]["wall_clock_minutes"] for e in ids]

    ax.bar(range(3), vals, color=["#e377c2", "#9467bd", "#ff7f0e"],
           edgecolor="black", linewidth=0.6, width=0.55)
    ax.set_xticks(range(3))
    ax.set_xticklabels([names[e] for e in ids], fontsize=9)
    ax.set_ylabel("Eval loss (lower is better)")

    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 3, 0.002)
    ax.set_ylim(lo - pad, hi + pad)

    for i, v in enumerate(vals):
        ax.text(i, v + pad * 0.06, f"{v:.6f}", ha="center", fontsize=9.5)

    diffs = []
    for i in (0, 1):
        d = vals[i] - vals[2]
        diffs.append(d)
        ax.text(i, vals[i] - pad * 0.14, f"vs E3: {d:+.6f}",
                ha="center", fontsize=8.5, color="0.25")

    sec = f"Training time — A1: {times[0]:.2f} min · A2: {times[1]:.2f} min · E3: {times[2]:.2f} min"
    ax.set_title(f"Ablation of importance-score components "
                 f"(all: Adaptive 50%, random order)\n{sec}", fontsize=11)
    add_seed_note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig5_adaptive_selection_ablation")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    exp = load_results()
    FIG_DIR.mkdir(exist_ok=True)

    generated: list[str] = []
    generated += fig1_eval_loss_comparison()
    generated += fig2_eval_loss_vs_time()
    generated += fig3_data_fraction_vs_eval_loss()
    generated += fig4_curriculum_comparison()
    generated += fig5_adaptive_selection_ablation()

    print("\nValidation:")
    ok = True
    expected = {f"fig{i}_{n}.{ext}"
                for i, n in [(1, "eval_loss_comparison"), (2, "eval_loss_vs_time"),
                             (3, "data_fraction_vs_eval_loss"), (4, "curriculum_comparison"),
                             (5, "adaptive_selection_ablation")]
                for ext in ("png", "pdf")}
    for name in sorted(expected):
        p = FIG_DIR / name
        size = p.stat().st_size if p.exists() else 0
        status = "OK" if size > 0 else "MISSING/EMPTY"
        if size == 0:
            ok = False
        print(f"  {p.name:<45} {size:>8,} bytes  {status}")

    print("\nAll 10 figures generated." if ok else "\nSOME FILES MISSING!")
    sys.exit(0 if ok else 1)
