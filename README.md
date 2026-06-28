💊 Medicine Recommendation System — Semantic Search over a Bangladeshi Medicine Catalog
Recommend medicines by what they treat, not just by name. Built with Sentence-Transformer embeddings on the MedEasy Bangladesh medicine dataset, versioned with DVC, and served through FastAPI.

Tech Stack
Sentence-Transformers (all-MiniLM-L6-v2, ONNX backend) — dense semantic embeddings
scikit-learn / NumPy — cosine similarity ranking on L2-normalized vectors
DVC — versioned, reproducible pipeline (data → features → model → metrics)
FastAPI — inference API
pandas, PyYAML — config-driven preprocessing

Project Structure
med-rec-mlsd/
├── configs/config.yaml          ← paths, API host/port, logging
├── params.yaml                  ← DVC-tracked hyperparameters
├── dvc.yaml / dvc.lock          ← pipeline definition
├── data/
│   ├── raw/                     ← medeasy_cleaned_database.csv (DVC-tracked)
│   ├── processed/                ← cleaned, text-concatenated rows
│   └── features/                ← embeddings.pkl, medicine_index.pkl, model_metadata.json
├── src/
│   ├── data/preprocess.py        ← Stage 1: clean + concatenate text fields
│   ├── features/generate_embeddings.py  ← Stage 2: MiniLM sentence embeddings
│   ├── features/build_features.py       ← legacy TF-IDF baseline (superseded)
│   ├── models/train.py           ← Stage 3: build + evaluate MedicineRecommender
│   └── serving/api.py            ← Stage 4: FastAPI inference server
├── models/recommender.pkl        ← serialized recommender (DVC output)
├── metrics/scores.json           ← evaluation metrics (DVC-tracked)
└── notebooks/01_eda.ipynb        ← exploratory data analysis

Setup
pip install -r requirements.txt
Raw data is DVC-tracked, not stored in git. Either run `dvc pull` (if you have access to the configured remote) or place medeasy_cleaned_database.csv under data/raw/ yourself before running the pipeline.

Run Pipeline
dvc repro                                    # runs all 3 active stages end-to-end
# or step by step:
python src/data/preprocess.py                # Step 1: clean + normalize raw CSV
python src/features/generate_embeddings.py   # Step 2: encode medicines with MiniLM
python src/models/train.py                   # Step 3: build recommender + write metrics/scores.json
dvc metrics show                             # view evaluation metrics

Launch API
uvicorn src.serving.api:app --reload --port 8000
Interactive docs: http://localhost:8000/docs

Example
POST /recommend
{"query": "fever headache pain", "top_k": 5}

→ Top 5 medicines ranked by cosine similarity, each with generic name, dosage form, manufacturer, price, and similarity score — plus a built-in disclaimer ("for informational purposes only, consult a physician").

Also available: POST /alternatives (similar medicines for a given medicine name) and GET /medicines/{medicine_name} (full record lookup).

Current Metrics (sample eval, 100 medicines × top-5 alternatives)
Model: sentence-transformers/all-MiniLM-L6-v2
Avg. similarity: 0.94
Catalog coverage: 17.4% of medicines appear at least once across the sampled recommendations
Catalog size: 2,376 medicines

Roadmap
- [ ] SHAP-based explainability for individual recommendations
- [ ] WandB experiment tracking across embedding/model iterations
- [ ] River/ADWIN-based drift detection in production
- [ ] Expanded Responsible-AI guardrails beyond the current disclaimer field

About
Semantic medicine-recommendation engine over a 2,300+ item Bangladeshi medicine catalog (MedEasy) — upgraded from a TF-IDF baseline to dense Sentence-Transformer embeddings, wrapped in a DVC-versioned pipeline and served via FastAPI.

Topics
nlp semantic-search sentence-transformers fastapi dvc mlops healthcare recommendation-system
