"""
scoring.py — compute importance scores for Alpaca examples.

Importance = α·diversity + β·complexity + γ·response_length_quality
           (α + β + γ = 1.0 by convention; defaults 0.333 each)

All sub-scores are normalised to [0, 1] before combining.

Usage
-----
    from src.scoring import score_dataset
    scored_ds = score_dataset(dataset, config)
"""
import numpy as np
from typing import Dict, Any, List
from src.utils import get_logger

logger = get_logger("scoring")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-score helpers
# ─────────────────────────────────────────────────────────────────────────────

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to [0, 1]. Handles degenerate constant arrays."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.ones_like(arr) * 0.5
    return (arr - lo) / (hi - lo)


def compute_complexity_scores(examples: List[Dict[str, str]]) -> np.ndarray:
    """
    Lightweight complexity proxy (no LLM required):
      - instruction character length (normalised)
      - approximate vocabulary richness: unique tokens / total tokens

    Both sub-signals are averaged, then the combined score is min-max normalised.

    DEITA uses a 7B–13B LLM-based scorer. We deliberately use a lightweight
    heuristic so it runs at sub-1B PEFT scale without an external API.
    This is stated explicitly as an adaptation in the paper Methodology section.
    """
    char_lengths = np.array([len(ex.get("instruction", "") + ex.get("input", ""))
                              for ex in examples], dtype=float)

    vocab_richness = []
    for ex in examples:
        text = (ex.get("instruction", "") + " " + ex.get("input", "")).lower().split()
        if len(text) == 0:
            vocab_richness.append(0.0)
        else:
            vocab_richness.append(len(set(text)) / len(text))
    vocab_richness = np.array(vocab_richness, dtype=float)

    complexity = (_minmax_norm(char_lengths) + _minmax_norm(vocab_richness)) / 2.0
    return _minmax_norm(complexity)


def compute_response_length_scores(examples: List[Dict[str, str]]) -> np.ndarray:
    """
    Response-length quality proxy: Alpaca responses that are too short (<20 chars)
    or extremely long (>2000 chars) are penalised; a mid-range band is favoured.

    This acts as a cheap quality filter — very short answers are likely low-information,
    very long ones may be padding. The score peaks at a band of roughly 200–800 chars.
    """
    lengths = np.array([len(ex.get("output", "")) for ex in examples], dtype=float)
    # Smooth bell-shaped reward centred on log-scale
    log_lengths = np.log1p(lengths)
    # Penalise extremes: score = 1 - |normalised_log_length - 0.6| / 0.6
    norm_log = _minmax_norm(log_lengths)
    target = 0.6  # empirically mid-range for Alpaca
    score = 1.0 - np.abs(norm_log - target) / max(target, 1 - target)
    score = np.clip(score, 0.0, 1.0)
    return score


def compute_diversity_scores(
    examples: List[Dict[str, str]],
    embedding_model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Diversity score = distance from each example's embedding to the dataset centroid.
    Examples that are far from the mean are more 'diverse' (complement the rest).

    Uses sentence-transformers; falls back to a TF-IDF cosine distance if
    sentence-transformers is unavailable (e.g., no internet).
    """
    texts = [
        (ex.get("instruction", "") + " " + ex.get("input", "")).strip()
        for ex in examples
    ]

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Computing embeddings with {embedding_model_name} ...")
        model = SentenceTransformer(embedding_model_name)
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                                  convert_to_numpy=True, normalize_embeddings=True)
        centroid = embeddings.mean(axis=0)
        # Cosine similarity: since embeddings are L2-normalised, dot product = cosine sim
        similarities = embeddings @ centroid
        diversity = 1.0 - similarities   # high similarity → low diversity
        return _minmax_norm(diversity)

    except Exception as e:
        logger.warning(f"sentence-transformers failed ({e}); falling back to TF-IDF diversity.")
        return _tfidf_diversity_fallback(texts)


def _tfidf_diversity_fallback(texts: List[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    tfidf = vec.fit_transform(texts).toarray()
    centroid = tfidf.mean(axis=0, keepdims=True)
    sims = cosine_similarity(tfidf, centroid).flatten()
    diversity = 1.0 - sims
    return _minmax_norm(diversity)


# ─────────────────────────────────────────────────────────────────────────────
# Main scoring entry-point
# ─────────────────────────────────────────────────────────────────────────────

def score_dataset(
    examples: List[Dict[str, str]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compute and attach importance scores to each example.

    Returns
    -------
    List of dicts with original fields + added keys:
      _score_diversity, _score_complexity, _score_response_length, _score_importance
    """
    scoring_cfg = config.get("scoring", {})
    alpha = scoring_cfg.get("alpha", 0.333)
    beta  = scoring_cfg.get("beta",  0.333)
    gamma = scoring_cfg.get("gamma", 0.334)
    emb_model = scoring_cfg.get("embedding_model", "all-MiniLM-L6-v2")

    logger.info(f"Scoring {len(examples)} examples (a={alpha}, b={beta}, g={gamma})")

    div_scores   = compute_diversity_scores(examples, emb_model)
    comp_scores  = compute_complexity_scores(examples)
    resp_scores  = compute_response_length_scores(examples)

    importance = alpha * div_scores + beta * comp_scores + gamma * resp_scores

    scored = []
    for i, ex in enumerate(examples):
        scored.append({
            **ex,
            "_score_diversity":        float(div_scores[i]),
            "_score_complexity":       float(comp_scores[i]),
            "_score_response_length":  float(resp_scores[i]),
            "_score_importance":       float(importance[i]),
        })

    logger.info(f"Importance score stats — mean: {importance.mean():.4f}, "
                f"std: {importance.std():.4f}, "
                f"min: {importance.min():.4f}, max: {importance.max():.4f}")
    return scored
