"""
src/features/build_features.py
--------------------------------
Builds TF-IDF features for the Medicine Recommendation System.

Inputs:
    data/processed/medeasy_processed.csv   (combined_text, medicine_name columns)

Outputs:
    data/features/tfidf_matrix.pkl         (scipy sparse matrix, joblib)
    data/features/medicine_index.pkl       (dict {medicine_name: row_index}, joblib)
    data/features/vectorizer.pkl           (fitted TfidfVectorizer, joblib)

Config:  configs/config.yaml
Params:  params.yaml  (section: "features")
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_features")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    logger.info("Loaded YAML: %s", path)
    return data or {}


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_features(config_path: str = "configs/config.yaml",
                   params_path: str = "params.yaml") -> None:
    """End-to-end feature-building pipeline."""

    # ------------------------------------------------------------------
    # 1. Load config & params
    # ------------------------------------------------------------------
    logger.info("Loading configuration …")
    config = load_yaml(config_path)
    params_all = load_yaml(params_path)

    feat_params: dict = params_all.get("features", {})
    logger.info("Feature params: %s", feat_params)

    tfidf_max_features: int = int(feat_params["tfidf_max_features"])
    ngram_min: int = int(feat_params["ngram_range_min"])
    ngram_max: int = int(feat_params["ngram_range_max"])
    top_k: int = int(feat_params["top_k"])  # retained in scope for downstream use

    # Resolve paths from config (fall back to sensible defaults)
    data_cfg: dict = config.get("data", {})
    processed_csv: Path = Path(
        data_cfg.get("processed_csv", "data/processed/medeasy_processed.csv")
    )
    features_dir: Path = Path(
        data_cfg.get("features_dir", "data/features")
    )

    logger.info("Processed CSV  : %s", processed_csv)
    logger.info("Features dir   : %s", features_dir)
    logger.info("top_k (stored) : %d", top_k)

    # ------------------------------------------------------------------
    # 2. Load processed CSV
    # ------------------------------------------------------------------
    logger.info("Loading processed CSV …")
    if not processed_csv.exists():
        raise FileNotFoundError(f"Processed CSV not found: {processed_csv}")

    df = pd.read_csv(processed_csv)
    logger.info("DataFrame shape: %s", df.shape)

    required_cols = {"combined_text", "medicine_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # ------------------------------------------------------------------
    # 3. Fill remaining nulls in combined_text
    # ------------------------------------------------------------------
    null_before = df["combined_text"].isna().sum()
    df["combined_text"] = df["combined_text"].fillna("")
    logger.info("Filled %d null(s) in 'combined_text' with empty string.", null_before)

    texts: list[str] = df["combined_text"].tolist()
    logger.info("Total documents to vectorise: %d", len(texts))

    # ------------------------------------------------------------------
    # 4 & 5. Fit TfidfVectorizer and transform
    # ------------------------------------------------------------------
    logger.info(
        "Fitting TfidfVectorizer  (max_features=%d, ngram_range=(%d,%d)) …",
        tfidf_max_features, ngram_min, ngram_max,
    )
    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=(ngram_min, ngram_max),
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)   # scipy sparse CSR matrix

    logger.info("TF-IDF matrix shape : %s", tfidf_matrix.shape)
    logger.info("Vocabulary size     : %d", len(vectorizer.vocabulary_))

    # ------------------------------------------------------------------
    # 6. Build medicine_index  {medicine_name: row_index}
    # ------------------------------------------------------------------
    logger.info("Building medicine_index …")
    medicine_index: dict[str, int] = {
        name: idx for idx, name in enumerate(df["medicine_name"].tolist())
    }
    logger.info("medicine_index entries: %d", len(medicine_index))

    # ------------------------------------------------------------------
    # 7–9. Save artefacts
    # ------------------------------------------------------------------
    ensure_dir(features_dir)

    tfidf_matrix_path = features_dir / "tfidf_matrix.pkl"
    medicine_index_path = features_dir / "medicine_index.pkl"
    vectorizer_path = features_dir / "vectorizer.pkl"

    logger.info("Saving tfidf_matrix → %s", tfidf_matrix_path)
    joblib.dump(tfidf_matrix, tfidf_matrix_path)

    logger.info("Saving medicine_index → %s", medicine_index_path)
    joblib.dump(medicine_index, medicine_index_path)

    logger.info("Saving vectorizer → %s", vectorizer_path)
    joblib.dump(vectorizer, vectorizer_path)

    # ------------------------------------------------------------------
    # 10. Summary print
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Feature engineering complete")
    print("=" * 55)
    print(f"  Vocabulary size  : {len(vectorizer.vocabulary_):,}")
    print(f"  Matrix shape     : {tfidf_matrix.shape}")
    print(f"  Medicine entries : {len(medicine_index):,}")
    print(f"  Saved to         : {features_dir.resolve()}")
    print("=" * 55 + "\n")

    logger.info("build_features finished successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build TF-IDF features for the Medicine Recommendation System."
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

    build_features(config_path=args.config, params_path=args.params)