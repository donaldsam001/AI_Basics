def build_features(result: dict) -> list[float]:

    return [
        result["semantic_score"] / 100,
        result["required_skill_score"] / 100,
        result["preferred_skill_score"] / 100,
        result["experience_score"] / 100,
        result["retrieval_score"],
    ]