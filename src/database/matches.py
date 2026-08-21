"""Match repository interacting with Supabase database."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
from supabase import Client
from src.database.client import get_supabase_client

logger = logging.getLogger(__name__)


class MatchRepository:
    """Repository for match results operations in Supabase."""

    def __init__(self, client: Optional[Client] = None) -> None:
        self._client = client

    @property
    def client(self) -> Client:
        if self._client is None:
            client = get_supabase_client()
            if client is None:
                raise RuntimeError(
                    "Supabase client is not configured. Please set SUPABASE_URL and SUPABASE_KEY."
                )
            self._client = client
        return self._client

    def save_results(self, results: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Save match results to Supabase."""
        res = self.client.table("match_results").insert(results).execute()
        return res.data or []

    def get_job_results(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieve match results for a specific job_id sorted by rank."""
        res = (
            self.client.table("match_results")
            .select("*")
            .eq("job_id", job_id)
            .order("rank")
            .execute()
        )
        return res.data or []