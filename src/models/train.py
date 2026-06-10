"""
src/models/train.py
--------------------
Trains (instantiates + evaluates) the MedicineRecommender and persists it.

Inputs:
    data/features/tfidf_matrix.pkl          (scipy sparse CSR, shape 2544 × N)
    data/features/medicine_index.pkl        (dict {medicine_name: row_index})
    data/features/vectorizer.pkl            (fitted TfidfVectorizer)
    data/processed/medeasy_processed.csv    (metadata columns)

Outputs:
    models/recommender.pkl                  (full MedicineRecommender object)
    metrics/scores.json                     (evaluation metrics)

Config:  configs/config.yaml
Params:  params.yaml  (section: "model")
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Shared helpers  (mirrors build_features.py conventions)
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    """Load a YAML file; raise FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    logger.info("Loaded YAML: %s", path)
    return data or {}


def load_artifact(path: str | Path) -> Any:
    """Load a joblib artifact; raise FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    obj = joblib.load(path)
    logger.info("Loaded artifact: %s  [type=%s]", path, type(obj).__name__)
    return obj


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# MedicineRecommender
# ---------------------------------------------------------------------------

# Metadata columns surfaced in every recommendation dict.
_META_COLS = [
    "generic_name",
    "dosage_form",
    "manufacturer",
    "mrp",
    "discounted_price",
]


class MedicineRecommender:
    """
    Content-based medicine recommender backed by a TF-IDF matrix.

    Parameters
    ----------
    tfidf_matrix : scipy sparse matrix, shape (n_medicines, n_features)
        Pre-built TF-IDF feature matrix; one row per medicine.
    medicine_index : dict[str, int]
        Mapping from medicine_name → row index in tfidf_matrix.
    df : pd.DataFrame
        Processed CSV with metadata columns (generic_name, dosage_form, …).
        Row order must match tfidf_matrix row order.
    vectorizer : TfidfVectorizer
        Fitted vectorizer used to transform free-text queries.
    params : dict
        Model hyper-parameters loaded from params.yaml → "model" section.
        Expected keys: similarity_metric (str), min_score_threshold (float).
    """

    def __init__(
        self,
        tfidf_matrix,
        medicine_index: dict[str, int],
        df: pd.DataFrame,
        vectorizer,
        params: dict,
    ) -> None:
        self.tfidf_matrix = tfidf_matrix
        self.medicine_index = medicine_index
        self.df = df.reset_index(drop=True)   # guarantee positional alignment
        self.vectorizer = vectorizer
        self.params = params

        # Derived attributes
        self.similarity_metric: str = str(params.get("similarity_metric", "cosine")).lower()
        self.min_score_threshold: float = float(params.get("min_score_threshold", 0.05))

        # Reverse look-up: row_index → medicine_name
        self.reverse_index: dict[int, str] = {v: k for k, v in medicine_index.items()}

        logger.info(
            "MedicineRecommender initialised — %d medicines, "
            "metric=%s, threshold=%.3f",
            len(self.medicine_index),
            self.similarity_metric,
            self.min_score_threshold,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_meta(self, row_idx: int, score: float) -> dict:
        """
        Build a result dict from a DataFrame row index and similarity score.
        Missing metadata columns are filled with None gracefully.
        """
        medicine_name = self.reverse_index.get(row_idx, "unknown")
        row = self.df.iloc[row_idx]

        result: dict[str, Any] = {"medicine_name": medicine_name}
        for col in _META_COLS:
            result[col] = row[col] if col in self.df.columns else None
        result["similarity_score"] = round(float(score), 6)
        return result

    def _cosine_scores(self, query_vec) -> np.ndarray:
        """
        Return a 1-D array of cosine similarity scores between query_vec
        and every row in tfidf_matrix.
        """
        # cosine_similarity returns shape (1, n); flatten to (n,)
        scores: np.ndarray = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend_by_query(
        self, query_text: str, top_k: int = 10
    ) -> list[dict]:
        """
        Recommend medicines for a free-text symptom / condition query.

        Parameters
        ----------
        query_text : str
            User's search string (e.g. "fever headache sore throat").
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[dict]
            Ranked list of recommendation dicts, highest-scoring first.
            Each dict contains: medicine_name, generic_name, dosage_form,
            manufacturer, mrp, discounted_price, similarity_score.
        """
        if not query_text or not query_text.strip():
            logger.warning("recommend_by_query called with empty query_text.")
            return []

        query_vec = self.vectorizer.transform([query_text.strip()])
        scores = self._cosine_scores(query_vec)

        # Mask scores below threshold and rank
        above_thresh = np.where(scores >= self.min_score_threshold)[0]
        if len(above_thresh) == 0:
            logger.info("No results above threshold %.3f for query: %r",
                        self.min_score_threshold, query_text)
            return []

        ranked = above_thresh[np.argsort(scores[above_thresh])[::-1]]
        top_indices = ranked[:top_k]

        results = [self._row_to_meta(int(idx), scores[idx]) for idx in top_indices]
        logger.debug("recommend_by_query(%r) → %d results", query_text, len(results))
        return results

    def recommend_alternatives(
        self, medicine_name: str, top_k: int = 5
    ) -> list[dict]:
        """
        Find medicines most similar to a given medicine (i.e. alternatives).

        Parameters
        ----------
        medicine_name : str
            Exact name present in medicine_index.
        top_k : int
            Number of alternatives to return (the medicine itself is excluded).

        Returns
        -------
        list[dict]
            Ranked list of similar medicines (excluding the query medicine).

        Raises
        ------
        KeyError
            If medicine_name is not found in the index.
        """
        if medicine_name not in self.medicine_index:
            raise KeyError(
                f"Medicine '{medicine_name}' not found in medicine_index. "
                "Check spelling or run build_features again."
            )

        source_idx = self.medicine_index[medicine_name]
        source_vec = self.tfidf_matrix[source_idx]   # sparse row (1, N)

        scores = self._cosine_scores(source_vec)

        # Exclude the medicine itself
        scores[source_idx] = -1.0

        # Apply threshold
        above_thresh = np.where(scores >= self.min_score_threshold)[0]
        if len(above_thresh) == 0:
            logger.info("No alternatives above threshold for '%s'.", medicine_name)
            return []

        ranked = above_thresh[np.argsort(scores[above_thresh])[::-1]]
        top_indices = ranked[:top_k]

        results = [self._row_to_meta(int(idx), scores[idx]) for idx in top_indices]
        logger.debug("recommend_alternatives(%r) → %d results", medicine_name, len(results))
        return results

    def evaluate(self, sample_size: int = 100, top_k: int = 10) -> dict:
        """
        Estimate recommender quality over a random sample of medicines.

        Metrics
        -------
        coverage : float
            Fraction of the *full* catalogue that appears in at least one
            recommendation across all sampled queries (0 – 1).
        avg_similarity : float
            Mean of the top-1 similarity score for each sampled query.
        total_medicines : int
            Total number of medicines in the index.

        Parameters
        ----------
        sample_size : int
            Number of medicines to sample for evaluation (default 100).
        top_k : int
            Candidates retrieved per sample query (default 10).

        Returns
        -------
        dict with keys: coverage, avg_similarity, total_medicines.
        """
        all_names = list(self.medicine_index.keys())
        total = len(all_names)
        sample_size = min(sample_size, total)

        sampled_names = random.sample(all_names, sample_size)
        logger.info("Evaluating on %d sampled medicines …", sample_size)

        recommended_set: set[str] = set()
        top1_scores: list[float] = []

        for name in sampled_names:
            try:
                results = self.recommend_alternatives(name, top_k=top_k)
            except KeyError:
                continue

            if results:
                top1_scores.append(results[0]["similarity_score"])
                for r in results:
                    recommended_set.add(r["medicine_name"])

        coverage = round(len(recommended_set) / total, 6) if total > 0 else 0.0
        avg_similarity = round(float(np.mean(top1_scores)), 6) if top1_scores else 0.0

        metrics = {
            "coverage": coverage,
            "avg_similarity": avg_similarity,
            "total_medicines": total,
            "sample_size": sample_size,
            "unique_recommended": len(recommended_set),
        }
        logger.info("Evaluation metrics: %s", metrics)
        return metrics


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train(
    config_path: str = "configs/config.yaml",
    params_path: str = "params.yaml",
) -> None:
    """Load inputs, build recommender, evaluate, and persist artefacts."""

    # ------------------------------------------------------------------
    # 1. Load config & params
    # ------------------------------------------------------------------
    logger.info("Loading configuration …")
    config = load_yaml(config_path)
    params_all = load_yaml(params_path)

    model_params: dict = params_all.get("model", {})
    feat_params: dict = params_all.get("features", {})
    logger.info("Model params : %s", model_params)

    # Resolve directories from config with safe defaults
    data_cfg: dict = config.get("data", {})
    paths_cfg: dict = config.get("paths", {})

    features_dir = Path(data_cfg.get("features_dir", "data/features"))
    processed_csv = Path(data_cfg.get("processed_csv", "data/processed/medeasy_processed.csv"))
    models_dir = Path(paths_cfg.get("models_dir", "models"))
    metrics_dir = Path(paths_cfg.get("metrics_dir", "metrics"))

    top_k: int = int(feat_params.get("top_k", 10))

    # ------------------------------------------------------------------
    # 2. Load feature artefacts
    # ------------------------------------------------------------------
    logger.info("Loading feature artefacts …")
    tfidf_matrix = load_artifact(features_dir / "tfidf_matrix.pkl")
    medicine_index: dict[str, int] = load_artifact(features_dir / "medicine_index.pkl")
    vectorizer = load_artifact(features_dir / "vectorizer.pkl")

    logger.info("TF-IDF matrix shape  : %s", tfidf_matrix.shape)
    logger.info("Medicine index size  : %d", len(medicine_index))

    # ------------------------------------------------------------------
    # 3. Load processed CSV (metadata)
    # ------------------------------------------------------------------
    logger.info("Loading processed CSV …")
    if not processed_csv.exists():
        raise FileNotFoundError(f"Processed CSV not found: {processed_csv}")

    df = pd.read_csv(processed_csv)
    logger.info("DataFrame shape: %s", df.shape)

    # Sanity-check alignment
    if len(df) != tfidf_matrix.shape[0]:
        raise ValueError(
            f"Row count mismatch: CSV has {len(df)} rows but "
            f"tfidf_matrix has {tfidf_matrix.shape[0]} rows. "
            "Re-run build_features to regenerate aligned artefacts."
        )

    # ------------------------------------------------------------------
    # 4. Instantiate MedicineRecommender
    # ------------------------------------------------------------------
    logger.info("Instantiating MedicineRecommender …")
    recommender = MedicineRecommender(
        tfidf_matrix=tfidf_matrix,
        medicine_index=medicine_index,
        df=df,
        vectorizer=vectorizer,
        params=model_params,
    )

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    logger.info("Running evaluation …")
    metrics = recommender.evaluate(sample_size=100, top_k=top_k)

    ensure_dir(metrics_dir)
    scores_path = metrics_dir / "scores.json"
    with scores_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved → %s", scores_path)

    # ------------------------------------------------------------------
    # 6. Save recommender
    # ------------------------------------------------------------------
    ensure_dir(models_dir)
    recommender_path = models_dir / "recommender.pkl"
    joblib.dump(recommender, recommender_path)
    logger.info("Recommender saved → %s", recommender_path)

    # ------------------------------------------------------------------
    # 7. Sanity-check: sample recommendation for "fever headache"
    # ------------------------------------------------------------------
    _print_sample_recommendations(recommender, top_k=top_k)

    # Final summary
    print("\n" + "=" * 60)
    print("  Training complete")
    print("=" * 60)
    print(f"  Total medicines     : {metrics['total_medicines']:,}")
    print(f"  Coverage            : {metrics['coverage']:.2%}")
    print(f"  Avg top-1 similarity: {metrics['avg_similarity']:.4f}")
    print(f"  Recommender saved   : {recommender_path.resolve()}")
    print(f"  Metrics saved       : {scores_path.resolve()}")
    print("=" * 60 + "\n")


def _print_sample_recommendations(
    recommender: MedicineRecommender, top_k: int = 5
) -> None:
    """Print a formatted table of recommendations for a canned query."""
    sample_query = "fever headache"
    logger.info("Sanity check — recommend_by_query(%r) …", sample_query)

    results = recommender.recommend_by_query(sample_query, top_k=top_k)

    print("\n" + "-" * 60)
    print(f"  Sample recommendations for query: '{sample_query}'")
    print("-" * 60)

    if not results:
        print("  (no results above threshold — check min_score_threshold)")
    else:
        for rank, r in enumerate(results, start=1):
            print(
                f"  {rank:>2}. {r['medicine_name']:<35} "
                f"score={r['similarity_score']:.4f}"
            )
            if r.get("generic_name"):
                print(f"       Generic : {r['generic_name']}")
            if r.get("dosage_form"):
                print(f"       Form    : {r['dosage_form']}")
            if r.get("mrp") is not None:
                print(f"       MRP     : {r['mrp']}  |  "
                      f"Discounted: {r.get('discounted_price', 'N/A')}")
            print()
    print("-" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the MedicineRecommender for the Medicine Recommendation System."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configs/config.yaml (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Path to params.yaml (default: params.yaml)",
    )
    args = parser.parse_args()

    try:
        train(config_path=args.config, params_path=args.params)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)