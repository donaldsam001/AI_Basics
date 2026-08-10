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
