# Repository Evaluation & Technical Debt Report: AI_Basic

**Date:** September 6, 2026  
**Repository:** `donaldsam001/AI_Basics`  
**Current Branch:** `develop`  
**Target Architecture:** CV–Job Semantic Matching & Reranking Service (SentenceTransformers + FAISS + XGBoost + Qwen LLM + FastAPI)

---

## 1. Executive Summary

`AI_Basic` is an end-to-end intelligent CV-to-Job matching pipeline combining multi-modal document extraction (PDF/DOCX), semantic embedding retrieval (`sentence-transformers/all-mpnet-base-v2` + FAISS), supervised candidate reranking (XGBoost), and LLM-based RAG explanations (Qwen).

While the core matching algorithms and feature engineering concepts are mathematically solid, **the repository currently suffers from significant architectural fragmentation, environment pollution, dual API divergence, and event-loop blocking issues.**

### Key Metrics & Findings
| Area | Status | Critical Issue |
| :--- | :--- | :--- |
| **API Architecture** | 🔴 Severe | Two competing FastAPI applications (`app/main.py` vs `src/api/app.py`). `Dockerfile` runs the legacy/outdated one. |
| **Python Environment** | 🔴 Severe | Root `venv/` is broken (`_posixsubprocess` missing); nested `extract_data/.venv` leaked into `$PATH`. |
| **Git & Storage** | 🟠 High | ~18MB of binary models, FAISS indices, and embedding NumPy arrays tracked directly in Git history. |
| **FastAPI Performance** | 🟠 High | Synchronous CPU-bound embedding inference runs directly inside `async def` routes, blocking the event loop. |
| **Security & Config** | 🟡 Medium | Developer-specific local absolute paths in `.env` (`/home/donaldsam/...`), wildcard CORS, incomplete `.gitignore`. |
| **Code Duplication** | 🟡 Medium | Redundant preprocessing modules, duplicate CLIs, multiple feature builder implementations, and stray empty files. |
| **ML Data Contract** | 🟡 Medium | XGBoost target `shortlisted` is synthetically derived from the input features (heuristic memorization). |

---

## 2. Detailed Problem Breakdown

---

### 🔴 Problem 1: Dual Divergent FastAPI Applications

The repository contains **two completely separate FastAPI services** that do not import or know about each other:

1. **`app/main.py`** (`app = FastAPI(...)`):
   - Defined endpoints: `GET /health`, `POST /predict`, `POST /match`, `POST /match/inline`.
   - Uses `model_pipeline.joblib` via `app/model.py`.
   - **Referenced by `Dockerfile` line 20:** `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`.
2. **`src/api/app.py`** (`create_app()`):
   - Defined endpoints: `GET /health`, `GET /ready`, `POST /api/v1/search`, `POST /api/v1/match`, `POST /api/v1/match/upload`, `POST /api/v1/explain`, `POST /api/v1/candidates/score`, `POST /api/v1/candidates/rank`.
   - Uses `MatchingService` and `ModelRegistry` in `src/services/`.
   - Used by new test suites (`tests/test_api_upload.py`).

**Impact:**
- Any container built with the existing `Dockerfile` **does not expose** the new `/api/v1/*` endpoints (candidate scoring, ranking, or file upload matching).
- Maintenance overhead: Schemas and endpoints are maintained in duplicate locations (`app/schemas.py` vs `src/api/schemas.py`).

---

### 🔴 Problem 2: Corrupted & Nested Virtual Environments

1. **Corrupted Root `venv`:**
   Running python or pip in `./venv/bin/pip` fails immediately:
   ```
   ModuleNotFoundError: No module named '_posixsubprocess'
   ```
   Checking `venv/pyvenv.cfg` reveals why:
   ```ini
   command = /home/donaldsam/Downloads/AI_Basic/extract_data/.venv/bin/python -m venv /home/donaldsam/Downloads/AI_Basic/venv
   ```
   The root `venv` was spawned from a nested virtual environment inside `extract_data/.venv`.
2. **Subdirectory Virtualenv:**
   `extract_data/.venv` exists inside a project subfolder. This directory leaked into the developer's global shell `$PATH`:
   `/home/donaldsam/Downloads/AI_Basic/extract_data/.venv/bin: ...`
3. **Missing Testing Dependencies in `requirements.txt`:**
   `requirements.txt` lacks `pytest`, `pytest-cov`, `httpx` (required by FastAPI `TestClient`), and `python-dotenv`.
4. **Python Version Inconsistency:**
   `Dockerfile` specifies `python:3.11-slim`, but local environment tools are running against `Python 3.14`, which creates C-extension incompatibilities with packages like `faiss-cpu`, `xgboost`, and `torch`.

---

### 🟠 Problem 3: Repository Bloat & Large Binary Artifacts in Git

The repository tracks large binary files, datasets, and generated embeddings directly in Git:

| File | Size | Problem |
| :--- | :--- | :--- |
| `data/embeddings/cv_embeddings.npy` | **11.96 MB** | Raw vector binary array; creates massive Git history bloat |
| `data/ats_resume_dataset_elite_v3.csv` | **4.96 MB** | Large raw CSV dataset checked into source control |
| `data/models/cv_job_xgb.json` | **485 KB** | Serialized model file |
| `data/models/xgb_pipeline.joblib` | **485 KB** | Serialized scikit-learn pipeline |
| `model_pipeline.joblib` (root) | **410 KB** | Duplicate legacy serialized pipeline |
| `data/embeddings/cv_embeddings.json` | **287 KB** | Vector metadata JSON |
| `data/faiss/cv.index` | **3 KB** | Binary FAISS index |

**Impact:**
- Cloning the repo downloads megabytes of ephemeral matrices and models.
- Any retraining or re-embedding produces binary diffs that permanently inflate repository size.
- Models should be pulled via an artifact store, Hugging Face Hub, DVC, or S3/GCS bucket during CI/CD build.

---

### 🟠 Problem 4: FastAPI Event Loop Blocking (Async Anti-Pattern)

In [`src/api/routes/candidates.py`](file:///home/donaldsam/Downloads/AI_Basic/src/api/routes/candidates.py#L31-L96):

```python
@router.post("/score", response_model=ScoreResponse)
async def score_candidate(request: Request, body: ScoreRequest):
    ...
    candidate, _ = (service.evaluate_cv(body.resume_text, job, explain=True), 0)
```
and:
```python
@router.post("/rank", response_model=List[ScoreResponse])
async def rank_candidates(request: Request, body: RankRequest):
    ...
    for cand in body.candidates:
        res, _ = service.evaluate_cv(cand.resume_text, job, explain=False), 0
```

**Impact:**
- `service.evaluate_cv` is heavily CPU-bound (it generates MPNet sentence embeddings and calculates cosine similarities).
- In FastAPI, declaring an endpoint as `async def` means it executes directly on the single-threaded asyncio event loop.
- Running CPU-bound loops without `fastapi.concurrency.run_in_threadpool` freezes the entire server; no other incoming HTTP requests can be accepted until the loop finishes.

---

### 🟡 Problem 5: Architectural Fragmentation & Code Duplication

1. **Duplicate Text Preprocessing:**
   - [`text_preprocessing/text_preprocessing.py`](file:///home/donaldsam/Downloads/AI_Basic/text_preprocessing/text_preprocessing.py): Uses `nltk` (`word_tokenize`, `lemmatizer`, `stopwords`).
   - [`src/preprocessing/cv_preprocessor.py`](file:///home/donaldsam/Downloads/AI_Basic/src/preprocessing/cv_preprocessor.py) & `common.py`: Uses custom regex and standalone rules.
   - Result: CVs preprocessed via `main.py` will have different tokens and lemma representations than CVs preprocessed via `src.preprocessing`.
2. **Duplicate CLI & Entry Points:**
   - Root `main.py`: Preprocesses PDFs in `example_data/pdf` to CSV.
   - `src/main.py`: CLI for running the matching system with `argparse`.
   - `run_intelligent_matching.py`: Third runner script in root with overlapping functionality.
3. **Duplicate Feature Engineering:**
   - `src/features/feature_engineering.py`
   - `src/models/feature_builder.py`
   - `src/ml/features.py` (backward-compatibility shim)
   - `train.py` (inline feature manipulation in pandas)
4. **Stray / Empty Files:**
   - `src/app.py`: Empty 1-byte file (contains only 2 empty lines).
   - Untracked files in git (`tests/test_api_upload.py`, `src/api/routes/`, etc.) need cleanup and staging.

---

### 🟡 Problem 6: Security, Secrets & Environment Hardcoding

1. **Absolute Local Paths in `.env`:**
   ```ini
   QWEN_MODEL_PATH=/home/donaldsam/data/models/qwen/Qwen3-0.6B-Q8_0.gguf
   ```
   This breaks on any container, remote VM, or teammate's computer.
2. **Incomplete `.gitignore`:**
   Current `.gitignore`:
   ```gitignore
   .venv/
   __pycache__/
   *.pyc
   .ipynb_checkpoints/
   .env
   .vscode/
   .idea/
   ```
   **Missing entries:** `venv/`, `*.npy`, `*.joblib`, `*.index`, `*.gguf`, `*.bin`, `.pytest_cache/`, `data/embeddings/`, `data/faiss/`.
3. **Permissive CORS:**
   In `app/main.py` line 125: `allow_origins=["*"]`.

---

### 🟡 Problem 7: Synthetic Target Leakage in ML Model Training

In `src/models/train_xgboost.py`:
```python
"label_provenance": (
    "The target 'shortlisted' is ALGORITHMICALLY derived from features "
    "like skill_match_score + experience_match + education_match. "
    "This model learns to reproduce the source scoring rule."
)
```
The supervised XGBoost model is trained on targets that were computed from a deterministic linear equation of the exact input features.
- The model is not learning human recruiter preferences or real hiring success.
- It is learning to approximate a simple weighted sum formula that could be computed deterministically in microseconds with 0 latency and no machine learning overhead.

---

## 3. Recommended Remediation Roadmap

### Phase 1: Environment & Repository Cleanup (Immediate)
- [ ] **Delete nested venvs:** Remove `extract_data/.venv` and remove it from system `$PATH`.
- [ ] **Rebuild a clean virtualenv** using Python 3.11 (`pyenv local 3.11.12 && python -m venv .venv`).
- [ ] **Update `.gitignore`** to ignore `venv/`, `*.joblib`, `*.npy`, `*.index`, `*.gguf`, and `.pytest_cache`.
- [ ] **Update `requirements.txt`** to include `pytest`, `httpx`, and `python-dotenv`.
- [ ] **Delete dead files:** Remove `src/app.py` (empty file).

### Phase 2: Unify FastAPI Architecture
- [ ] **Designate `src/api/app.py` as the canonical production FastAPI application.**
- [ ] Port any useful endpoints from `app/main.py` (such as `POST /match/inline` or `/predict`) into `src/api/routes/` as sub-routers.
- [ ] Update `Dockerfile`:
  ```dockerfile
  CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- [ ] Retire or deprecate the redundant `app/` directory.

### Phase 3: Performance & Event Loop Fixes
- [ ] In `src/api/routes/candidates.py`, wrap CPU-bound operations in `run_in_threadpool`:
  ```python
  from fastapi.concurrency import run_in_threadpool

  # Inside async def score_candidate:
  candidate, _ = await run_in_threadpool(service.evaluate_cv, body.resume_text, job, True)
  ```
- [ ] For `rank_candidates`, batch vector embeddings together rather than looping single items sequentially.

### Phase 4: Module Consolidation & Package Setup
- [ ] Add `pyproject.toml` to define standard packaging, pytest configuration, and package dependencies so tests can be run with a simple `pytest` command without setting `PYTHONPATH`.
- [ ] Consolidate `text_preprocessing/` into `src/preprocessing/`.
- [ ] Consolidate `train.py` and `src/models/train_xgboost.py` into a unified training pipeline.

---

*Report generated by Antigravity Assistant.*
