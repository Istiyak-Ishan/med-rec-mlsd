import os
import yaml
import logging
import pandas as pd
from pathlib import Path


# -----------------------------
# YAML loader
# -----------------------------
def load_yaml(file_path):
    """Safely load YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------
# Main pipeline
# -----------------------------
def main():

    # -----------------------------
    # Logging setup
    # -----------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        # -----------------------------
        # Project root-safe config path
        # -----------------------------
        BASE_DIR = Path(__file__).resolve().parents[2]
        CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
        PARAMS_PATH = BASE_DIR / "params.yaml"

        # -----------------------------
        # Load configs
        # -----------------------------
        logger.info("Loading configuration and parameters...")
        config = load_yaml(CONFIG_PATH)
        params = load_yaml(PARAMS_PATH)

        # -----------------------------
        # Correct nested config access
        # -----------------------------
        raw_data_path = BASE_DIR / config["paths"]["raw_data_path"]
        processed_data_path = BASE_DIR / config["paths"]["processed_data_path"]

        prep_params = params["preprocessing"]
        text_cols = prep_params["text_cols"]
        fill_value = prep_params["fill_value"]

        # -----------------------------
        # Load raw dataset
        # -----------------------------
        logger.info(f"Loading raw data from {raw_data_path}...")
        df = pd.read_csv(raw_data_path)

        # -----------------------------
        # Clean text columns
        # -----------------------------
        logger.info("Cleaning text columns...")
        for col in text_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna(fill_value)
                    .replace("Information not provided.", "")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
            else:
                logger.warning(f"Missing text column: {col}")

        # -----------------------------
        # Numeric columns
        # -----------------------------
        logger.info("Processing numeric columns...")
        for col in ["mrp", "discounted_price"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
            else:
                logger.warning(f"Missing numeric column: {col}")

        # -----------------------------
        # Drop invalid rows
        # -----------------------------
        logger.info("Dropping invalid rows...")
        if "medicine_name" in df.columns:
            df = df.dropna(subset=["medicine_name"])
        else:
            raise ValueError("'medicine_name' column is required but missing")

        # -----------------------------
        # Feature engineering: combined_text
        # -----------------------------
        logger.info("Creating combined_text feature...")

        combine_cols = [
            "generic_name",
            "indications",
            "mode_of_action",
            "side_effects"
        ]

        for col in combine_cols:
            if col not in df.columns:
                df[col] = ""
                logger.warning(f"Missing column filled with empty string: {col}")

        df["combined_text"] = (
            df[combine_cols]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # -----------------------------
        # Save output
        # -----------------------------
        logger.info(f"Saving processed data to {processed_data_path}...")

        os.makedirs(processed_data_path.parent, exist_ok=True)
        df.to_csv(processed_data_path, index=False)

        # -----------------------------
        # Final output
        # -----------------------------
        print(f"Preprocessing complete ✔ Shape: {df.shape[0]} x {df.shape[1]}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


# -----------------------------
if __name__ == "__main__":
    main()