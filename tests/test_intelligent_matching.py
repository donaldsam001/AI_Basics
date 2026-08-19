"""Unit tests for the 9-step CV-Job Intelligent Matching System."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.intelligent_matching import parse_resume, CVJobIntelligentMatchingSystem


class IntelligentMatchingTests(unittest.TestCase):
    def test_parse_resume_raw_text(self):
        text = "Alice Johnson. Senior Python Developer with 5 years experience in Django and PostgreSQL."
        parsed = parse_resume(text)
        self.assertEqual(parsed["raw_text"], text)
        self.assertIn("python", [s.casefold() for s in parsed.get("technical_skills", []) + parsed.get("programming_languages", [])])

    def test_end_to_end_matching(self):
        cv1 = parse_resume("Alice Developer. 6 years experience in Python, FastApi, PostgreSQL, Docker, AWS.")
        cv1["candidate_name"] = "Alice Developer"

        cv2 = parse_resume("Bob Designer. 1 year experience in Photoshop, Illustrator, Figma.")
        cv2["candidate_name"] = "Bob Designer"

        job = {
            "job_title": "Senior Backend Engineer",
            "job_description": "Senior Backend Engineer required with 5+ years experience in Python, PostgreSQL, Docker.",
            "required_skills": ["python", "postgresql", "docker"],
            "years_of_experience": 5,
        }

        system = CVJobIntelligentMatchingSystem()
        output = system.process_matching([cv1, cv2], job, top_k_retrieval=2, top_k_explain=0)

        self.assertEqual(output["job_title"], "Senior Backend Engineer")
        self.assertEqual(len(output["rankings"]), 2)
        top_rank = output["rankings"][0]
        self.assertEqual(top_rank["candidate_name"], "Alice Developer")
        self.assertIn("skill_gap_analysis", top_rank)


if __name__ == "__main__":
    unittest.main()
