"""Minimal usage example: python example_resume_parser.py"""

from __future__ import annotations

import json

from resume_parser import parse_resume


if __name__ == "__main__":
    resume_text = """
    Jane Doe
    Senior Backend Engineer
    jane@example.com

    EXPERIENCE
    Senior Backend Engineer | Acme Technologies | Jan 2021 - Present
    - Built Spring Boot microservices with Java, PostgreSQL, Docker, and AWS.

    EDUCATION
    Example University
    Bachelor of Science in Computer Science | 2016 - 2020 | GPA: 3.8/4.0

    CERTIFICATIONS
    AWS Certified Developer - Associate | Issued by Amazon Web Services | Jun 2023

    PROJECTS
    Resume Matcher
    Role: Developer
    - Built a Python and FastAPI matching service using scikit-learn.
    """
    print(json.dumps(parse_resume(resume_text), indent=2, ensure_ascii=False))
