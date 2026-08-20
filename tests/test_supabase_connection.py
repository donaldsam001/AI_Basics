"""Tests for Supabase connection, client initialization, and repository interactions."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from src.database.candidates import CandidateRepository
from src.database.client import check_supabase_connection, get_supabase_client, init_supabase
from src.database.jobs import JobRepository
from src.database.matches import MatchRepository


class SupabaseClientTests(unittest.TestCase):
    """Unit tests for Supabase client creation and connection checking."""

    def test_get_supabase_client_without_credentials_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            client = get_supabase_client()
            self.assertIsNone(client)

    def test_init_supabase_without_credentials_raises_value_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                init_supabase()
            self.assertIn("Supabase credentials missing", str(ctx.exception))

    @patch("src.database.client.create_client")
    def test_get_supabase_client_with_credentials_returns_client(self, mock_create_client):
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        client = get_supabase_client(url="https://test.supabase.co", key="test-key")
        self.assertEqual(client, mock_client)
        mock_create_client.assert_called_once_with("https://test.supabase.co", "test-key")

    @patch("src.database.client.get_supabase_client")
    def test_check_supabase_connection_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.limit.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[])
        mock_get_client.return_value = mock_client

        is_connected = check_supabase_connection()
        self.assertTrue(is_connected)

    @patch("src.database.client.get_supabase_client")
    def test_check_supabase_connection_failure(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.table.side_effect = Exception("Connection error")
        mock_get_client.return_value = mock_client

        is_connected = check_supabase_connection()
        self.assertFalse(is_connected)


class CandidateRepositoryTests(unittest.TestCase):
    """Unit tests for CandidateRepository using mock Supabase client."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_table = MagicMock()
        self.mock_client.table.return_value = self.mock_table
        self.repo = CandidateRepository(client=self.mock_client)

    def test_get_all(self):
        self.mock_table.select.return_value.execute.return_value.data = [{"id": "1", "name": "Alice"}]
        candidates = self.repo.get_all()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["name"], "Alice")
        self.mock_client.table.assert_called_with("candidates")

    def test_get_by_id(self):
        self.mock_table.select.return_value.eq.return_value.execute.return_value.data = [
            {"id": "1", "name": "Alice"}
        ]
        candidate = self.repo.get_by_id("1")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["name"], "Alice")

    def test_create(self):
        new_cand = {"name": "Bob"}
        self.mock_table.insert.return_value.execute.return_value.data = [{"id": "2", "name": "Bob"}]
        created = self.repo.create(new_cand)
        self.assertEqual(created[0]["id"], "2")
        self.mock_table.insert.assert_called_once_with(new_cand)

    def test_update(self):
        updates = {"name": "Bob Updated"}
        self.mock_table.update.return_value.eq.return_value.execute.return_value.data = [
            {"id": "2", "name": "Bob Updated"}
        ]
        updated = self.repo.update("2", updates)
        self.assertEqual(updated[0]["name"], "Bob Updated")

    def test_delete(self):
        self.mock_table.delete.return_value.eq.return_value.execute.return_value.data = [
            {"id": "2"}
        ]
        deleted = self.repo.delete("2")
        self.assertEqual(len(deleted), 1)


class JobRepositoryTests(unittest.TestCase):
    """Unit tests for JobRepository using mock Supabase client."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_table = MagicMock()
        self.mock_client.table.return_value = self.mock_table
        self.repo = JobRepository(client=self.mock_client)

    def test_get_all(self):
        self.mock_table.select.return_value.execute.return_value.data = [{"id": "j1", "title": "Developer"}]
        jobs = self.repo.get_all()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["title"], "Developer")

    def test_get_by_id(self):
        self.mock_table.select.return_value.eq.return_value.execute.return_value.data = [
            {"id": "j1", "title": "Developer"}
        ]
        job = self.repo.get_by_id("j1")
        self.assertIsNotNone(job)
        self.assertEqual(job["id"], "j1")


class MatchRepositoryTests(unittest.TestCase):
    """Unit tests for MatchRepository using mock Supabase client."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_table = MagicMock()
        self.mock_client.table.return_value = self.mock_table
        self.repo = MatchRepository(client=self.mock_client)

    def test_save_results(self):
        results = [{"job_id": "j1", "candidate_id": "c1", "rank": 1}]
        self.mock_table.insert.return_value.execute.return_value.data = results
        saved = self.repo.save_results(results)
        self.assertEqual(len(saved), 1)

    def test_get_job_results(self):
        self.mock_table.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [
            {"job_id": "j1", "candidate_id": "c1", "rank": 1}
        ]
        res = self.repo.get_job_results("j1")
        self.assertEqual(len(res), 1)


class LiveSupabaseIntegrationTests(unittest.TestCase):
    """Live integration test connecting to configured Supabase server."""

    def test_live_supabase_connection(self):
        client = get_supabase_client()
        if client is None:
            self.skipTest("Supabase credentials not configured in environment.")

        # Test live connectivity using check_supabase_connection
        is_connected = check_supabase_connection(client)
        self.assertTrue(is_connected, "Failed to connect to live Supabase database.")


if __name__ == "__main__":
    unittest.main()
