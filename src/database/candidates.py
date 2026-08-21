"""Candidate repository interacting with Supabase database."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from supabase import Client
from src.database.client import get_supabase_client

logger = logging.getLogger(__name__)


class CandidateRepository:
    """Repository for candidate CRUD operations in Supabase."""

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

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all candidates from Supabase."""
        res = self.client.table("candidates").select("*").execute()
        return res.data or []

    def get_by_id(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a candidate by ID."""
        res = self.client.table("candidates").select("*").eq("id", candidate_id).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]
        return None

    def create(self, candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Insert a new candidate into Supabase."""
        res = self.client.table("candidates").insert(candidate).execute()
        return res.data or []

    def update(self, candidate_id: str, updates: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update an existing candidate."""
        res = self.client.table("candidates").update(updates).eq("id", candidate_id).execute()
        return res.data or []

    def delete(self, candidate_id: str) -> List[Dict[str, Any]]:
        """Delete a candidate by ID."""
        res = self.client.table("candidates").delete().eq("id", candidate_id).execute()
        return res.data or []