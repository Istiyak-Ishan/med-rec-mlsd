"""
generate_embeddings.py
Stage 2 (replaces build_features.py): Encode medicines with MiniLM sentence transformer.
Outputs: embeddings.pkl (np.ndarray), medicine_index.pkl (dict), model_metadata.json
"""

import json
import logging
import sys
from pathlib import Path

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


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def build_medicine_index(df: pd.DataFrame) -> dict[str, int]:
    """Map medicine_name → row integer index."""
    return {name: idx for idx, name in enumerate(df["medicine_name"].tolist())}


def generate_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    """
    Load MiniLM and encode all texts.
    Returns float32 numpy array of shape (n_medicines, embedding_dim).
    """
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, backend='onnx')

    logger.info(f"Encoding {len(texts)} medicines in batches of {batch_size}…")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,   # L2-norm → cosine sim = dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    logger.info(f"Embedding matrix shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def main() -> None:
    config = load_config()
    params = load_params()
    feat_params = params["features"]

    # ── Paths ──────────────────────────────────────────────────────────────
    processed_path: str = config["paths"]["processed_data_path"]
    features_dir = Path(config["paths"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = features_dir / "embeddings.pkl"
    index_path      = features_dir / "medicine_index.pkl"
    metadata_path   = features_dir / "model_metadata.json"

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info(f"Loading processed data from {processed_path}")
    df = pd.read_csv(processed_path)
    logger.info(f"Loaded {len(df)} rows")

    if "combined_text" not in df.columns:
        raise ValueError("'combined_text' column missing — run preprocess stage first.")

    texts: list[str] = df["combined_text"].fillna("").tolist()

    # ── Generate embeddings ────────────────────────────────────────────────
    model_name: str  = feat_params["model_name"]
    batch_size: int  = feat_params["batch_size"]
    normalize: bool  = feat_params["normalize_embeddings"]

    embeddings = generate_embeddings(texts, model_name, batch_size, normalize)

    # ── Build medicine index ───────────────────────────────────────────────
    medicine_index: dict[str, int] = build_medicine_index(df)
    logger.info(f"Medicine index built: {len(medicine_index)} entries")

    # ── Save outputs ───────────────────────────────────────────────────────
    joblib.dump(embeddings, embeddings_path)
    logger.info(f"Saved embeddings → {embeddings_path}  [{embeddings.nbytes / 1e6:.1f} MB]")

    joblib.dump(medicine_index, index_path)
    logger.info(f"Saved medicine_index → {index_path}")

    metadata = {
        "model_name": model_name,
        "embedding_dim": embeddings.shape[1],
        "n_medicines": embeddings.shape[0],
        "normalized": normalize,
        "batch_size": batch_size,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata → {metadata_path}")

    logger.info("generate_embeddings stage complete.")
    logger.info(f"  Shape : {embeddings.shape}")
    logger.info(f"  dtype : {embeddings.dtype}")
    logger.info(f"  Model : {model_name}")


if __name__ == "__main__":
    main()