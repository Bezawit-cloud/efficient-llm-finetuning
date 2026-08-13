"""
select_and_order.py — data selection and curriculum ordering.

Selection methods
-----------------
  "full"     — use all examples (E1)
  "random"   — random fraction (E2, E5)
  "adaptive" — top-k% by importance score (E3, E4, ablations)

Ordering methods
----------------
  "random"     — shuffle (E1, E2, E3, ablations)
  "curriculum" — sort selected subset easy→hard by complexity sub-score (E4, E5)

Usage
-----
    from src.select_and_order import select_and_order
    subset = select_and_order(scored_examples, config)
"""
import random as _random
from typing import Any, Dict, List
from src.utils import get_logger

logger = get_logger("select_and_order")


def select_and_order(
    scored_examples: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Apply selection then ordering to a scored dataset.

    Parameters
    ----------
    scored_examples : output of score_dataset() — each item has _score_* fields.
    config          : merged experiment config dict.

    Returns
    -------
    Ordered list of selected examples (scoring fields stripped out).
    """
    data_cfg = config.get("data", {})
    sel_cfg  = data_cfg.get("selection", {})
    method   = sel_cfg.get("method", "full")
    fraction = float(sel_cfg.get("fraction", 1.0))
    ordering = data_cfg.get("ordering", "random")
    seed     = config.get("seed", 42)

    n_total = len(scored_examples)
    logger.info(f"Selection: method={method}, fraction={fraction}, n_total={n_total}")

    # ── 1. Selection ──────────────────────────────────────────────────────────
    if method == "full":
        selected = list(scored_examples)

    elif method == "random":
        rng = _random.Random(seed)
        k = max(1, int(n_total * fraction))
        selected = rng.sample(scored_examples, k)

    elif method == "adaptive":
        # Top-k% by importance score (DEITA-inspired)
        k = max(1, int(n_total * fraction))
        selected = sorted(scored_examples,
                          key=lambda x: x["_score_importance"],
                          reverse=True)[:k]

    else:
        raise ValueError(f"Unknown selection method: {method!r}")

    logger.info(f"Selected {len(selected)} / {n_total} examples ({100*len(selected)/n_total:.1f}%)")

    # ── 2. Ordering ───────────────────────────────────────────────────────────
    if ordering == "random":
        rng = _random.Random(seed + 1)   # different seed from selection
        _random.Random(seed + 1).shuffle(selected)

    elif ordering == "curriculum":
        # Easy → Hard: sort ascending by complexity sub-score
        # Curriculum signal = complexity sub-score (same metric as scoring — no second signal)
        selected = sorted(selected, key=lambda x: x.get("_score_complexity", 0.0))
        logger.info("Curriculum ordering applied: easy→hard by complexity sub-score")

    else:
        raise ValueError(f"Unknown ordering method: {ordering!r}")

    # ── 3. Strip internal scoring fields before returning ─────────────────────
    score_keys = {"_score_diversity", "_score_complexity", "_score_response_length", "_score_importance"}
    clean = [{k: v for k, v in ex.items() if k not in score_keys} for ex in selected]

    return clean


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: produce ALL five experiment variants at once
# ─────────────────────────────────────────────────────────────────────────────

def produce_all_variants(
    scored_examples: List[Dict[str, Any]],
    base_config: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience wrapper used on Day 5 to pre-produce all 5 dataset variants.
    Returns a dict keyed by experiment ID.
    """
    from copy import deepcopy

    variants = {
        "E1": {"data": {"selection": {"method": "full",     "fraction": 1.0}, "ordering": "random"}},
        "E2": {"data": {"selection": {"method": "random",   "fraction": 0.5}, "ordering": "random"}},
        "E3": {"data": {"selection": {"method": "adaptive", "fraction": 0.5}, "ordering": "random"}},
        "E4": {"data": {"selection": {"method": "adaptive", "fraction": 0.5}, "ordering": "curriculum"}},
        "E5": {"data": {"selection": {"method": "random",   "fraction": 0.5}, "ordering": "curriculum"}},
    }

    results = {}
    for exp_id, override in variants.items():
        cfg = deepcopy(base_config)
        # simple merge override
        cfg["data"]["selection"] = override["data"]["selection"]
        cfg["data"]["ordering"]  = override["data"]["ordering"]
        logger.info(f"Producing variant {exp_id} ...")
        results[exp_id] = select_and_order(scored_examples, cfg)
        logger.info(f"  {exp_id}: {len(results[exp_id])} examples")

    return results
