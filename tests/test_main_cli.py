"""CLI behavior tests that do not load the embedding model."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from src import main as cli


class MainCliTests(unittest.TestCase):
    def test_default_command_uses_baseline_ranking_without_faiss(self):
        matcher = MagicMock()
        matcher.rank_candidates.return_value = [
            {"candidate_name": "Ada", "final_score": 0.9},
        ]

        with (
            patch.object(sys, "argv", ["src.main"]),
            patch.object(cli, "load_cvs", return_value=[{"candidate_name": "Ada"}]),
            patch.object(cli, "load_job", return_value={"job_title": "Engineer"}),
            patch.object(cli, "CVJobMatcher", return_value=matcher),
        ):
            cli.main()

        matcher.rank_candidates.assert_called_once()
        matcher.rank_retrieved_candidates.assert_not_called()

    def test_faiss_paths_must_be_provided_together(self):
        with (
            patch.object(sys, "argv", ["src.main", "--faiss-index", "data/faiss/cv.index"]),
            patch.object(cli, "load_cvs", return_value=[]),
            patch.object(cli, "load_job", return_value={"job_title": "Engineer"}),
        ):
            with self.assertRaises(SystemExit) as error:
                cli.main()

        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
