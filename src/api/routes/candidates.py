from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Request, HTTPException

router = APIRouter(prefix="/api/v1/candidates", tags=["Candidates"])

class ScoreRequest(BaseModel):
    candidate_id: str
    job_id: str
    resume_text: str
    job_description: str

class ScoreResponse(BaseModel):
    candidate_id: str
    job_id: str
    score: float
    recommendation: str
    confidence: float
    feature_summary: Dict[str, float]
    model_version: str

class CandidateInput(BaseModel):
    candidate_id: str
    resume_text: str

class RankRequest(BaseModel):
    job_id: str
    job_description: str
    candidates: List[CandidateInput]

@router.post("/score", response_model=ScoreResponse)
async def score_candidate(request: Request, body: ScoreRequest):
    service = request.app.state.service
    job_data = {"job_title": "", "job_description": body.job_description, "required_skills": ""}
    job = service.build_job(job_data)
    
    try:
        # Evaluate CV returns candidate details including score
        candidate, _ = await service.evaluate_cv_async(body.resume_text, job, explain=True) if hasattr(service, "evaluate_cv_async") else (service.evaluate_cv(body.resume_text, job, explain=True), 0)
        
        # Format properly
        score = candidate.get("xgb_probability", candidate.get("match_probability", 0.0))
        recommendation = candidate.get("recommendation", "REVIEW")
        
        # Build feature summary
        features = {}
        if "features" in candidate:
            for k in ["skill_coverage_ratio", "experience_score", "semantic_similarity"]:
                if k in candidate["features"]:
                    features[k] = candidate["features"][k]
                    
        return ScoreResponse(
            candidate_id=body.candidate_id,
            job_id=body.job_id,
            score=score,
            recommendation=recommendation.upper(),
            confidence=score,
            feature_summary=features,
            model_version="1.0"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/rank", response_model=List[ScoreResponse])
async def rank_candidates(request: Request, body: RankRequest):
    service = request.app.state.service
    job_data = {"job_title": "", "job_description": body.job_description, "required_skills": ""}
    job = service.build_job(job_data)
    
    results = []
    for cand in body.candidates:
        try:
            res, _ = service.evaluate_cv(cand.resume_text, job, explain=False), 0
            score = res.get("xgb_probability", res.get("match_probability", 0.0))
            recommendation = res.get("recommendation", "REVIEW")
            features = {}
            if "features" in res:
                for k in ["skill_coverage_ratio", "experience_score", "semantic_similarity"]:
                    if k in res["features"]:
                        features[k] = res["features"][k]
            
            results.append(ScoreResponse(
                candidate_id=cand.candidate_id,
                job_id=body.job_id,
                score=score,
                recommendation=recommendation.upper(),
                confidence=score,
                feature_summary=features,
                model_version="1.0"
            ))
        except Exception as exc:
            pass # Skip failed
            
    # Sort descending
    results.sort(key=lambda x: x.score, reverse=True)
    return results
