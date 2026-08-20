"""Database package providing Supabase integration and repository models."""

from src.database.candidates import CandidateRepository
from src.database.client import check_supabase_connection, get_supabase_client, init_supabase
from src.database.jobs import JobRepository
from src.database.matches import MatchRepository

__all__ = [
    "get_supabase_client",
    "init_supabase",
    "check_supabase_connection",
    "CandidateRepository",
    "JobRepository",
    "MatchRepository",
]
