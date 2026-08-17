# CV-Job Semantic Matching

An explainable, CPU-oriented CV ranking system using `sentence-transformers/all-mpnet-base-v2`.

## Architecture

Existing `extract_data/` reads PDF and DOCX files. Existing `resume_parser/` turns CV text into structured fields. The new `src/preprocessing/` creates focused semantic text, `src/embeddings/` owns one lazily loaded CPU model, and `src/matching/` combines semantic relevance with explicit skills and experience.

`all-mpnet-base-v2` provides strong general-purpose sentence embeddings. Text is encoded with normalized vectors and batch size 8, so cosine similarity is a dot product and CPU memory use stays modest. The model downloads on its first embedding request and is cached by Hugging Face locally.

## Install and run

```bash
python -m pip install -r requirements.txt
python -m src.main --cv data/cvs.csv --job data/jobs.csv
```

CV input requires `candidate_name,cv_text`; job input requires `job_title,job_description`. The CLI also accepts this repository's `raw_text` CV column and bundled `Job Title`, `Required Skills`, and `Experience Years` job columns. It ranks all CVs against the first job row; use `--job-index` to select another zero-based row.

## Matching

CV semantic text emphasizes experience, skills, education, certifications, and projects. It excludes contact data and candidate name. Job semantic text emphasizes title, requirements, responsibilities, qualifications, and technical keywords.

The default transparent score is: semantic relevance 50%, required skills 30%, preferred skills 10%, and experience 10%. Skill aliases such as `K8s`/`Kubernetes` and `Postgres`/`PostgreSQL` are normalized in `src/matching/similarity.py`; weights and recommendation thresholds are configurable in `src/matching/matcher.py`.

The resulting `semantic_match_score` supports comparing relevance between candidates. **Embedding similarity is not a hiring probability**, and this tool is not an autonomous hiring decision system.

## CPU notes

The model is loaded once per matcher and candidate embeddings are generated in batches of eight. A job is encoded once per ranking operation, while NumPy dot products compare it with all CV vectors. CLI runs persist source-hashed `data/embeddings/cv_embeddings.npy` and `job_embeddings.npy`; only changed source text is encoded again.

## XGBoost Candidate Reranking

FAISS remains the fast semantic retrieval layer: it uses MPNet embeddings to find only the top `--retrieval-k` CVs. XGBoost is an optional supervised scoring layer applied **only** to those retrieved CVs, so it does not replace FAISS or run across the entire candidate collection.

The feature module has numeric candidate features (years of experience, portfolio presence, parsed skill count, raw-text length, and an ordinal education level) plus reusable CV--JD comparison features (semantic score, required/preferred skill coverage, experience, education, and title similarity). Skills are lowercased, deduplicated, and normalized for common aliases. FAISS similarity is reported separately from the XGBoost probability; cosine similarity is not a probability.

Train the included CV-level model:

```bash
python -m src.models.train_xgboost \
  --dataset example_data/ml_resume_dataset_4500.csv \
  --output data/models/cv_job_xgb.json
```

Training uses an 80/20 stratified split and reports class distribution, accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, and a classification report. It saves the native model, ordered feature schema, and sorted feature importance to `data/models/cv_job_xgb.json`, `data/models/xgb_features.json`, and `data/models/xgb_feature_importance.csv`.

Then retrieve and rerank candidates:

```bash
python -m src.main \
  --cv data/preprocess/preprocessed_cvs.csv \
  --job example_data/jd/job_roles_IT_filtered.csv \
  --faiss-index data/faiss/cv.index \
  --faiss-metadata data/faiss/cv_metadata.json \
  --retrieval-k 50 \
  --xgb-model data/models/cv_job_xgb.json
```

`ml_resume_dataset_4500.csv` labels individual CV records: `1` means suitable and `0` means unsuitable according to that dataset's source semantics. It does not include CV--JD pairs or a job identifier, so the provided model is a CV-level suitability model, not a calibrated job-specific suitability probability. The pipeline deliberately does not fabricate pair labels; add a pair-labelled dataset before training the comparison features for job-specific probability estimates.
