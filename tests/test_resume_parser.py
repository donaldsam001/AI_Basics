"""Smoke tests for the public parser API (standard-library unittest only)."""

from __future__ import annotations

import unittest

from resume_parser import parse_resume


class ResumeParserTests(unittest.TestCase):
    def test_missing_values_have_safe_defaults(self) -> None:
        parsed = parse_resume("")
        self.assertIsNone(parsed["candidate_name"])
        self.assertEqual(parsed["years_of_experience"], 0)
        self.assertEqual(parsed["technical_skills"], [])
        self.assertEqual(parsed["education"], [])

    def test_dictionary_matching_uses_canonical_values(self) -> None:
        parsed = parse_resume("Skills: JS, SpringBoot, K8s, postgres")
        self.assertEqual(parsed["programming_languages"], ["JavaScript"])
        self.assertEqual(parsed["frameworks"], ["Spring Boot"])
        self.assertEqual(parsed["databases"], ["PostgreSQL"])
        self.assertEqual(parsed["devops_tools"], ["Kubernetes"])

    def test_dates_gpa_and_work_history(self) -> None:
        text = """
        Ada Lovelace
        EXPERIENCE
        Software Engineer | Example Corp | Feb 2020 - Dec 2022
        - Built REST APIs.

        EDUCATION
        Example University
        Bachelor of Science in Computing | 2016 - 2020 | GPA: 3.75/4.0
        """
        parsed = parse_resume(text)
        self.assertEqual(parsed["candidate_name"], "Ada Lovelace")
        self.assertEqual(parsed["current_position"], "Software Engineer")
        self.assertEqual(parsed["gpa"], "3.75")
        self.assertEqual(parsed["work_experience"][0]["start_date"], "2020-02")
        self.assertEqual(parsed["education"][0]["major"], "Computing")


if __name__ == "__main__":
    unittest.main()
