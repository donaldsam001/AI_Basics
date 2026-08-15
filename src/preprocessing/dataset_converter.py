"""
Convert dataset.csv into the schema used by preprocessed_cvs.csv.

Input columns:
    id
    resume_id
    resume_text
    resume_skills
    experience_years
    education_level
    projects
    job_role
    cleaned_text

Output columns:
    candidate_id
    file_name
    file_type
    raw_text
    cleaned_text
    candidate_name
    current_position
    years_of_experience
    education
    gpa
    technical_skills
    soft_skills
    programming_languages
    frameworks
    libraries
    databases
    cloud_platforms
    devops_tools
    operating_systems
    testing_tools
    certifications
    projects
    work_experience
    achievements
    industries
    domains
    spoken_languages
    keywords
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_COLUMNS = [
    "candidate_id",
    "file_name",
    "file_type",
    "raw_text",
    "cleaned_text",
    "candidate_name",
    "current_position",
    "years_of_experience",
    "education",
    "gpa",
    "technical_skills",
    "soft_skills",
    "programming_languages",
    "frameworks",
    "libraries",
    "databases",
    "cloud_platforms",
    "devops_tools",
    "operating_systems",
    "testing_tools",
    "certifications",
    "projects",
    "work_experience",
    "achievements",
    "industries",
    "domains",
    "spoken_languages",
    "keywords",
]


REQUIRED_INPUT_COLUMNS = [
    "resume_id",
    "resume_text",
    "resume_skills",
    "experience_years",
    "education_level",
    "projects",
    "job_role",
    "cleaned_text",
]


# ---------------------------------------------------------------------------
# Skill dictionaries
# ---------------------------------------------------------------------------

PROGRAMMING_LANGUAGES = {
    "python",
    "java",
    "javascript",
    "typescript",
    "c",
    "c++",
    "c#",
    "go",
    "golang",
    "rust",
    "kotlin",
    "swift",
    "php",
    "ruby",
    "scala",
    "r",
    "matlab",
    "perl",
    "dart",
    "lua",
    "bash",
    "shell",
    "sql",
    "pl/sql",
}

FRAMEWORKS = {
    "spring",
    "spring boot",
    "springboot",
    "django",
    "flask",
    "fastapi",
    "react",
    "reactjs",
    "next.js",
    "nextjs",
    "angular",
    "vue",
    "vue.js",
    "express",
    "express.js",
    "node.js",
    "nodejs",
    "laravel",
    "rails",
    "ruby on rails",
    ".net",
    "asp.net",
    "tensorflow",
    "pytorch",
}

LIBRARIES = {
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "sklearn",
    "opencv",
    "beautifulsoup",
    "beautifulsoup4",
    "requests",
    "axios",
    "jquery",
    "huggingface",
    "transformers",
}

DATABASES = {
    "mysql",
    "postgresql",
    "postgres",
    "sqlite",
    "mongodb",
    "mongo",
    "redis",
    "oracle",
    "oracle database",
    "mariadb",
    "sql server",
    "mssql",
    "firebase",
    "dynamodb",
    "elasticsearch",
    "neo4j",
}

CLOUD_PLATFORMS = {
    "aws",
    "amazon web services",
    "azure",
    "microsoft azure",
    "gcp",
    "google cloud",
    "google cloud platform",
    "heroku",
    "vercel",
    "netlify",
    "digitalocean",
}

DEVOPS_TOOLS = {
    "docker",
    "docker compose",
    "kubernetes",
    "k8s",
    "jenkins",
    "gitlab ci",
    "github actions",
    "circleci",
    "terraform",
    "ansible",
    "prometheus",
    "grafana",
    "nginx",
    "apache",
    "git",
    "gitlab",
    "github",
}

OPERATING_SYSTEMS = {
    "windows",
    "windows 10",
    "windows 11",
    "linux",
    "ubuntu",
    "debian",
    "centos",
    "red hat",
    "rhel",
    "fedora",
    "macos",
    "mac os",
    "unix",
    "kali linux",
}

TESTING_TOOLS = {
    "junit",
    "pytest",
    "unittest",
    "selenium",
    "cypress",
    "playwright",
    "postman",
    "newman",
    "jest",
    "mocha",
    "mockito",
}

SOFT_SKILLS = {
    "communication",
    "teamwork",
    "leadership",
    "problem solving",
    "problem-solving",
    "critical thinking",
    "time management",
    "adaptability",
    "adaptability",
    "collaboration",
    "creativity",
    "presentation",
    "negotiation",
    "decision making",
    "decision-making",
}

CERTIFICATION_KEYWORDS = {
    "aws certified",
    "azure certification",
    "google cloud certification",
    "comptia",
    "cisco certified",
    "ccna",
    "ccnp",
    "pmp",
    "scrum master",
    "oracle certified",
    "microsoft certified",
    "certified kubernetes",
}

# Generic technical skills that do not fit the more specific categories.
GENERIC_TECHNICAL_SKILLS = {
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data analysis",
    "data analytics",
    "excel",
    "power bi",
    "tableau",
    "computer vision",
    "natural language processing",
    "nlp",
    "cybersecurity",
    "networking",
    "network security",
    "cloud computing",
    "software development",
    "software engineering",
    "web development",
    "mobile development",
    "database management",
    "api",
    "rest api",
    "restful api",
    "microservices",
    "object oriented programming",
    "oop",
    "data structures",
    "algorithms",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def is_empty(value: Any) -> bool:
    """Return True if a value should be treated as empty."""
    if value is None:
        return True

    if isinstance(value, float) and pd.isna(value):
        return True

    text = str(value).strip()

    return text == "" or text.lower() in {
        "nan",
        "none",
        "null",
        "n/a",
        "na",
    }


def clean_text(value: Any) -> str:
    """Normalize a text value."""
    if is_empty(value):
        return ""

    text = str(value)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_skill(skill: str) -> str:
    """Normalize a skill while keeping a readable representation."""
    skill = clean_text(skill)

    if not skill:
        return ""

    return skill.strip(" ,;|[](){}\"'")


def parse_list(value: Any) -> list[str]:
    """
    Parse values such as:

        Python, Java, Docker
        ["Python", "Java", "Docker"]
        ['Python', 'Java']
        Python; Java; Docker

    into a clean list.
    """
    if is_empty(value):
        return []

    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()

        # Try Python/JSON list syntax first.
        if (
            (text.startswith("[") and text.endswith("]"))
            or (text.startswith("(") and text.endswith(")"))
        ):
            try:
                parsed = ast.literal_eval(text)

                if isinstance(parsed, (list, tuple, set)):
                    items = list(parsed)
                else:
                    items = [text]
            except (ValueError, SyntaxError):
                try:
                    parsed = json.loads(text)

                    if isinstance(parsed, list):
                        items = parsed
                    else:
                        items = [text]
                except (ValueError, json.JSONDecodeError):
                    items = re.split(r"[,;|]", text)
        else:
            items = re.split(r"[,;|\n]", text)

    result = []

    for item in items:
        item = normalize_skill(str(item))

        if item:
            result.append(item)

    return unique_preserve_order(result)


def unique_preserve_order(items: list[str]) -> list[str]:
    """Remove duplicates without changing order."""
    result = []
    seen = set()

    for item in items:
        key = item.lower().strip()

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


def normalize_lookup(value: str) -> str:
    """Normalize a value for dictionary matching."""
    value = value.lower().strip()

    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)

    return value


# ---------------------------------------------------------------------------
# Skill classification
# ---------------------------------------------------------------------------

def classify_skills(skills: list[str]) -> dict[str, list[str]]:
    """
    Classify skills into the target schema.

    A skill can only be assigned to its most specific category.
    """

    result = {
        "technical_skills": [],
        "soft_skills": [],
        "programming_languages": [],
        "frameworks": [],
        "libraries": [],
        "databases": [],
        "cloud_platforms": [],
        "devops_tools": [],
        "operating_systems": [],
        "testing_tools": [],
        "certifications": [],
    }

    lookup_sets = {
        "programming_languages": PROGRAMMING_LANGUAGES,
        "frameworks": FRAMEWORKS,
        "libraries": LIBRARIES,
        "databases": DATABASES,
        "cloud_platforms": CLOUD_PLATFORMS,
        "devops_tools": DEVOPS_TOOLS,
        "operating_systems": OPERATING_SYSTEMS,
        "testing_tools": TESTING_TOOLS,
        "soft_skills": SOFT_SKILLS,
        "certifications": CERTIFICATION_KEYWORDS,
        "technical_skills": GENERIC_TECHNICAL_SKILLS,
    }

    for skill in skills:
        normalized = normalize_lookup(skill)

        if not normalized:
            continue

        matched = False

        # More specific categories first.
        category_order = [
            "programming_languages",
            "frameworks",
            "libraries",
            "databases",
            "cloud_platforms",
            "devops_tools",
            "operating_systems",
            "testing_tools",
            "certifications",
            "soft_skills",
            "technical_skills",
        ]

        for category in category_order:
            known_values = lookup_sets[category]

            # Exact match.
            if normalized in known_values:
                result[category].append(skill)
                matched = True
                break

            # Handle values such as:
            # "Python programming"
            # "Experience with Docker"
            if any(
                known in normalized
                for known in known_values
                if len(known) >= 3
            ):
                result[category].append(skill)
                matched = True
                break

        # Anything not recognized is still retained as a technical skill.
        if not matched:
            result["technical_skills"].append(skill)

    for key in result:
        result[key] = unique_preserve_order(result[key])

    return result


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

def normalize_projects(value: Any) -> list[str]:
    """
    Convert projects into a clean list.

    The source dataset does not necessarily contain structured project
    objects, so projects are preserved as text rather than invented.
    """
    return parse_list(value)


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

def build_keywords(
    job_role: str,
    education: str,
    skills: dict[str, list[str]],
    projects: list[str],
) -> list[str]:
    """Build a searchable keyword list."""
    keywords = []

    if job_role:
        keywords.append(job_role)

    if education:
        keywords.append(education)

    for category_values in skills.values():
        keywords.extend(category_values)

    keywords.extend(projects)

    return unique_preserve_order(keywords)


# ---------------------------------------------------------------------------
# Row conversion
# ---------------------------------------------------------------------------

def convert_row(row: pd.Series) -> dict[str, Any]:
    """Convert one dataset.csv row to the target schema."""

    candidate_id = clean_text(row.get("resume_id"))

    if not candidate_id:
        candidate_id = clean_text(row.get("id"))

    raw_text = clean_text(row.get("resume_text"))

    cleaned = clean_text(row.get("cleaned_text"))

    # If cleaned_text is unavailable, fall back to resume_text.
    if not cleaned:
        cleaned = raw_text

    job_role = clean_text(row.get("job_role"))
    education = clean_text(row.get("education_level"))

    years_of_experience = row.get("experience_years")

    if is_empty(years_of_experience):
        years_of_experience = ""

    # Convert numeric-looking experience values to a clean representation.
    else:
        try:
            number = float(years_of_experience)

            if number.is_integer():
                years_of_experience = int(number)
            else:
                years_of_experience = number
        except (TypeError, ValueError):
            years_of_experience = clean_text(years_of_experience)

    skills = parse_list(row.get("resume_skills"))
    classified = classify_skills(skills)

    projects = normalize_projects(row.get("projects"))

    keywords = build_keywords(
        job_role=job_role,
        education=education,
        skills=classified,
        projects=projects,
    )

    return {
        "candidate_id": candidate_id,

        # Source dataset does not contain the original file name.
        "file_name": "",

        # The source is a CSV dataset, not an individual CV file.
        "file_type": "csv",

        "raw_text": raw_text,
        "cleaned_text": cleaned,

        # Not available in the source dataset.
        "candidate_name": "",

        "current_position": job_role,
        "years_of_experience": years_of_experience,
        "education": education,

        # Not available in the source dataset.
        "gpa": "",

        "technical_skills": json.dumps(
            classified["technical_skills"],
            ensure_ascii=False,
        ),

        "soft_skills": json.dumps(
            classified["soft_skills"],
            ensure_ascii=False,
        ),

        "programming_languages": json.dumps(
            classified["programming_languages"],
            ensure_ascii=False,
        ),

        "frameworks": json.dumps(
            classified["frameworks"],
            ensure_ascii=False,
        ),

        "libraries": json.dumps(
            classified["libraries"],
            ensure_ascii=False,
        ),

        "databases": json.dumps(
            classified["databases"],
            ensure_ascii=False,
        ),

        "cloud_platforms": json.dumps(
            classified["cloud_platforms"],
            ensure_ascii=False,
        ),

        "devops_tools": json.dumps(
            classified["devops_tools"],
            ensure_ascii=False,
        ),

        "operating_systems": json.dumps(
            classified["operating_systems"],
            ensure_ascii=False,
        ),

        "testing_tools": json.dumps(
            classified["testing_tools"],
            ensure_ascii=False,
        ),

        "certifications": json.dumps(
            classified["certifications"],
            ensure_ascii=False,
        ),

        "projects": json.dumps(
            projects,
            ensure_ascii=False,
        ),

        # Not available as a separate structured field.
        "work_experience": "",

        "achievements": "",

        "industries": "",

        "domains": json.dumps(
            [job_role] if job_role else [],
            ensure_ascii=False,
        ),

        "spoken_languages": "",

        "keywords": json.dumps(
            keywords,
            ensure_ascii=False,
        ),
    }


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------

def convert_dataset(
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Convert the entire CSV dataset."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    df = pd.read_csv(
        input_path,
        encoding="utf-8",
        keep_default_na=False,
    )

    missing_columns = [
        column
        for column in REQUIRED_INPUT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Input dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    converted_rows = []

    for _, row in df.iterrows():
        converted_rows.append(convert_row(row))

    result = pd.DataFrame(
        converted_rows,
        columns=OUTPUT_COLUMNS,
    )

    # Ensure output directory exists.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert dataset.csv into the "
            "preprocessed_cvs.csv schema."
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        default="dataset.csv",
        help="Path to the input dataset.csv",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="preprocessed_dataset.csv",
        help="Path to the converted CSV",
    )

    args = parser.parse_args()

    result = convert_dataset(
        input_path=args.input,
        output_path=args.output,
    )

    print("=" * 60)
    print("Dataset conversion completed")
    print("=" * 60)
    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Rows  : {len(result)}")
    print(f"Cols  : {len(result.columns)}")
    print()
    print("Output columns:")
    for column in result.columns:
        print(f"  - {column}")


if __name__ == "__main__":
    main()