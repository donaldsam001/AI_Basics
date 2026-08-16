class CVRetriever:

    def __init__(
        self,
        index,
        metadata,
    ):
        self.index = index
        self.metadata = metadata

    def retrieve(
        self,
        job_embedding,
        top_k: int = 20,
    ):
        scores, indices = self.index.search(
            job_embedding.reshape(1, -1),
            top_k,
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):
            if idx < 0:
                continue

            candidate = self.metadata[idx]

            results.append({
                "candidate": candidate,
                "retrieval_score": float(score),
            })

        return results