"""Supabase database client initialization and connection helpers."""

from __future__ import annotations

import logging
import os
from typing import Optional

from supabase import Client, create_client

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


def get_supabase_client(
    url: Optional[str] = None,
    key: Optional[str] = None,
) -> Optional[Client]:
    """Create and return a Supabase client instance.

    If url or key are omitted, reads SUPABASE_URL and SUPABASE_KEY (or
    SUPABASE_SERVICE_ROLE_KEY) from the environment. Returns None if
    credentials are not configured.
    """
    url = url or os.environ.get("SUPABASE_URL")
    key = key or os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        logger.warning(
            "Supabase credentials incomplete (SUPABASE_URL=%s, SUPABASE_KEY configured=%s)",
            bool(url),
            bool(key),
        )
        return None

    try:
        return create_client(url, key)
    except Exception as exc:
        logger.error("Failed to initialize Supabase client: %s", exc)
        return None


def init_supabase(
    url: Optional[str] = None,
    key: Optional[str] = None,
) -> Client:
    """Initialize and return a Supabase client or raise ValueError if credentials missing."""
    client = get_supabase_client(url=url, key=key)
    if client is None:
        raise ValueError(
            "Supabase credentials missing. Please configure SUPABASE_URL and "
            "SUPABASE_KEY in your environment or .env file."
        )
    return client


def check_supabase_connection(client: Optional[Client] = None) -> bool:
    """Check whether Supabase is reachable and responding."""
    if client is None:
        client = get_supabase_client()
    if client is None:
        return False

    try:
        # Perform a lightweight query to test connectivity
        client.table("candidates").select("id").limit(1).execute()
        return True
    except Exception as exc:
        logger.warning("Supabase connection check failed: %s", exc)
        return False