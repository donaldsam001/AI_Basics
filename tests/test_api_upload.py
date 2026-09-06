"""Tests for /api/v1/match/upload endpoint with JSON, plain text, CSV, and file uploads."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import PipelineConfig
from src.services.matching_service import ModelRegistry, MatchingService


@pytest.fixture
def test_client():
    config = PipelineConfig()
    
    def mock_registry_factory(cfg):
        mock_reg = MagicMock(spec=ModelRegistry)
        mock_reg.readiness.return_value = {
            "ready": True,
            "core_ready": True,
            "llm_available": False,
            "components": {},
        }
        mock_reg.cvs = []
        return mock_reg

    app = create_app(config=config, registry_factory=mock_registry_factory)

    with TestClient(app) as client:
        # Pre-set app.state.service for testing
        mock_service = MagicMock(spec=MatchingService)
        mock_service.build_job.side_effect = lambda d: {
            "job_title": d.get("title") or "Parsed Job",
            "job_description": d.get("description", ""),
            "required_skills": d.get("required_skills", []),
        }
        mock_service.evaluate_cv.return_value = (
            {
                "candidate_id": "uploaded-cv",
                "candidate_name": "Test Candidate",
                "rank": 1,
                "match_probability": 0.85,
                "matched_skills": ["Python"],
                "missing_required_skills": [],
            },
            {"total_ms": 12.3},
        )
        app.state.service = mock_service
        yield client


def test_upload_json_job(test_client):
    cv_content = b"Senior Python Developer with 5 years experience in FastAPI and Docker."
    files = {"cv": ("resume.txt", cv_content, "text/plain")}
    data = {
        "job": '{"title": "Backend Engineer", "description": "Looking for Python developer", "required_skills": ["Python"]}'
    }
    
    response = test_client.post("/api/v1/match/upload", data=data, files=files)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["job_title"] == "Backend Engineer"


def test_upload_plain_text_job(test_client):
    cv_content = b"Data Analyst with SQL and Python expertise."
    files = {"cv": ("cv.txt", cv_content, "text/plain")}
    data = {
        "job": "Data Analyst Position\nRequire 3 years of SQL, Python, and Tableau experience."
    }

    response = test_client.post("/api/v1/match/upload", data=data, files=files)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["job_title"] == "Data Analyst Position"


def test_upload_csv_cv_and_job_file(test_client):
    cv_csv = b"skill,years\nPython,5\nFastAPI,3\nDocker,2"
    job_csv = b"Job Title,Description\nML Engineer,Build NLP models with PyTorch"
    
    files = {
        "cv": ("candidate.csv", cv_csv, "text/csv"),
        "job_file": ("job.csv", job_csv, "text/csv"),
    }

    response = test_client.post("/api/v1/match/upload", files=files)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["job_title"] != ""
