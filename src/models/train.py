"""
train.py
Stage 3: Build MedicineRecommender from MiniLM embeddings and evaluate.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Config helpers ──────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Recommender ─────────────────────────────────────────────────────────────

class MedicineRecommender:
    """
    Embedding-based medicine recommender using MiniLM sentence transformer.
    Similarity is computed as dot product of L2-normalized vectors (= cosine similarity).
    """

    def __init__(
        self,
        embeddings: np.ndarray,          # shape (n_medicines, 384), float32, L2-normalized
        medicine_index: dict[str, int],  # {medicine_name: row_index}
        df: pd.DataFrame,
        model_name: str,
        params: dict,
    ) -> None:
        self.embeddings      = embeddings
        self.medicine_index  = medicine_index
        self.df              = df.reset_index(drop=True)
        self.params          = params
        self.min_threshold   = params["model"]["min_score_threshold"]
        self.top_k_default   = params["features"].get("top_k", 10)

        # Reverse index: row_index → medicine_name
        self.reverse_index: dict[int, str] = {v: k for k, v in medicine_index.items()}

        self._model_name = model_name
        self.model = None   # loaded lazily — not serialized with joblib
        logger.info(
            f"Recommender ready — {len(medicine_index)} medicines, "
            f"embedding dim {embeddings.shape[1]}"
        )


    def _get_model(self) -> SentenceTransformer:
        """Load model on first use (not stored in pickle)."""
        if self.model is None:
            logger.info(f"Loading SentenceTransformer: {self._model_name}")
            self.model = SentenceTransformer(self._model_name, backend='onnx')
        return self.model

    # ── Internal helpers ────────────────────────────────────────────────────

    def _row_to_dict(self, row_idx: int, score: float) -> dict:
        """Convert a DataFrame row to the standard result dict."""
        row = self.df.iloc[row_idx]
        return {
            "medicine_name":    str(row.get("medicine_name", "")),
            "generic_name":     str(row.get("generic_name", "")),
            "dosage_form":      str(row.get("dosage_form", "")),
            "manufacturer":     str(row.get("manufacturer", "")),
            "mrp":              float(row.get("mrp", 0.0)),
            "discounted_price": float(row.get("discounted_price", 0.0)),
            "similarity_score": round(float(score), 4),
        }

    def _score_all(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a single query vector (1, 384)
        and the full embedding matrix (n, 384).
        Because both are L2-normalized, dot product == cosine similarity.
        """
        return (self.embeddings @ query_vec.T).flatten()

    # ── Public API ──────────────────────────────────────────────────────────

    def recommend_by_query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Encode free-text query and return top-k similar medicines."""
        if top_k is None:
            top_k = self.top_k_default

        query_vec = self._get_model().encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )                                        # shape (1, 384)

        scores = self._score_all(query_vec)        # shape (n_medicines,)

        # Filter and sort
        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked:
            score = float(scores[idx])
            if score < self.min_threshold:
                break
            results.append(self._row_to_dict(int(idx), score))
            if len(results) >= top_k:
                break

        logger.info(f"Query '{query_text[:50]}' → {len(results)} results")
        return results

    def recommend_alternatives(
        self,
        medicine_name: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Return top-k medicines most similar to the given medicine."""
        if top_k is None:
            top_k = 5

        if medicine_name not in self.medicine_index:
            raise KeyError(f"Medicine not found: '{medicine_name}'")

        row_idx = self.medicine_index[medicine_name]
        query_vec = self.embeddings[row_idx : row_idx + 1]   # shape (1, 384)

        scores = self._score_all(query_vec)

        # Exclude the medicine itself
        scores[row_idx] = -1.0

        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked:
            score = float(scores[idx])
            if score < self.min_threshold:
                break
            results.append(self._row_to_dict(int(idx), score))
            if len(results) >= top_k:
                break

        logger.info(f"Alternatives for '{medicine_name}' → {len(results)} results")
        return results

    def evaluate(self) -> dict:
        """
        Sample 100 random medicines, run recommend_alternatives for each,
        and compute coverage + average similarity score.
        """
        n_sample = min(100, len(self.medicine_index))
        sampled_names = random.sample(list(self.medicine_index.keys()), n_sample)

        appeared_in_recs: set[str] = set()
        all_scores: list[float] = []

        for name in sampled_names:
            try:
                recs = self.recommend_alternatives(name, top_k=5)
                for r in recs:
                    appeared_in_recs.add(r["medicine_name"])
                    all_scores.append(r["similarity_score"])
            except Exception as e:
                logger.warning(f"Skipped '{name}' during eval: {e}")

        total = len(self.medicine_index)
        coverage = len(appeared_in_recs) / total if total > 0 else 0.0
        avg_sim  = float(np.mean(all_scores)) if all_scores else 0.0

        return {
            "coverage":         round(coverage, 4),
            "avg_similarity":   round(avg_sim, 4),
            "total_medicines":  total,
            "sample_size":      n_sample,
            "model":            self._model_name,
        }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    params = load_params()

    features_dir  = Path(config["paths"]["features_dir"])
    models_dir    = Path(config["paths"]["models_dir"])
    metrics_dir   = Path(config["paths"]["metrics_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings (numpy, not scipy sparse) ──────────────────────────
    embeddings_path = features_dir / "embeddings.pkl"
    index_path      = features_dir / "medicine_index.pkl"
    metadata_path   = features_dir / "model_metadata.json"

    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings: np.ndarray = joblib.load(embeddings_path)

    logger.info(f"Loading medicine index from {index_path}")
    medicine_index: dict[str, int] = joblib.load(index_path)

    with open(metadata_path) as f:
        model_metadata = json.load(f)
    model_name: str = model_metadata["model_name"]

    # ── Load processed dataframe ───────────────────────────────────────────
    processed_path: str = config["paths"]["processed_data_path"]
    logger.info(f"Loading processed data from {processed_path}")
    df = pd.read_csv(processed_path)

    # ── Instantiate recommender ────────────────────────────────────────────
    recommender = MedicineRecommender(
        embeddings=embeddings,
        medicine_index=medicine_index,
        df=df,
        model_name=model_name,
        params=params,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    logger.info("Running evaluation…")
    scores = recommender.evaluate()
    logger.info(f"Evaluation scores: {scores}")

    scores_path = metrics_dir / "scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    logger.info(f"Metrics saved → {scores_path}")

    # ── Save recommender ───────────────────────────────────────────────────
    rec_path = models_dir / "recommender.pkl"
    # Unload model before pickling (ONNX InferenceSession is not serializable)
    recommender.model = None
    joblib.dump(recommender, rec_path)
    logger.info(f"Recommender saved → {rec_path}")

    # Reload for sanity check
    recommender.model = None  # will lazy-load on first call

    # ── Sanity check ───────────────────────────────────────────────────────
    logger.info("Sanity check: query = 'fever headache pain'")
    sample = recommender.recommend_by_query("fever headache pain", top_k=3)
    for r in sample:
        logger.info(
            f"  {r['medicine_name']}  "
            f"({r['generic_name']})  "
            f"score={r['similarity_score']}"
        )


if __name__ == "__main__":
    main()