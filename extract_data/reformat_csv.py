import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


INPUT_FILE = Path("/home/donaldsam/Downloads/AI_Basic/Train_Resume_Data.csv")
OUTPUT_FILE = Path("Train_Resume_Data_parsed.csv")


def clean_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null"}:
        return None

    return value


def split_values(value: Any) -> list[str]:
    value = clean_value(value)

    if not value:
        return []

    # Handle values such as:
    # ['Python', 'Java', 'SQL']
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]
        except (ValueError, SyntaxError):
            pass

    return [
        item.strip()
        for item in re.split(r"[,;|]", value)
        if item.strip()
    ]


def unique_sorted(values: list[str]) -> list[str]:
    result = {}

    for value in values:
        value = value.strip()

        if value:
            result[value.lower()] = value

    return sorted(result.values(), key=str.lower)


def classify_skills(skills: list[str]) -> dict[str, list[str]]:

    programming_languages = {
        "python", "java", "javascript", "typescript",
        "c", "c++", "c#", "go", "golang", "rust",
        "php", "ruby", "kotlin", "swift", "scala", "r"
    }

    frameworks = {
        "spring", "spring boot", "django", "flask",
        "fastapi", "react", "reactjs", "nextjs",
        "next.js", "angular", "vue", "vue.js",
        "express", "tensorflow", "pytorch", "keras"
    }

    libraries = {
        "numpy", "pandas", "matplotlib", "scikit-learn",
        "sklearn", "requests", "axios", "lodash"
    }

    databases = {
        "mysql", "postgresql", "postgres", "mongodb",
        "sqlite", "oracle", "redis", "mssql",
        "sql server"
    }

    cloud_platforms = {
        "aws", "amazon web services", "azure",
        "gcp", "google cloud", "google cloud platform",
        "firebase"
    }

    devops_tools = {
        "docker", "kubernetes", "jenkins",
        "gitlab ci", "github actions", "terraform",
        "ansible", "git"
    }

    operating_systems = {
        "linux", "ubuntu", "windows", "macos", "unix"
    }

    testing_tools = {
        "junit", "pytest", "selenium",
        "postman", "cypress", "playwright"
    }

    technical = []
    programming = []
    framework_list = []
    library_list = []
    database_list = []
    cloud = []
    devops = []
    operating_system = []
    testing = []

    for skill in skills:

        normalized = skill.lower().strip()

        if normalized in programming_languages:
            programming.append(skill)

        elif normalized in frameworks:
            framework_list.append(skill)

        elif normalized in libraries:
            library_list.append(skill)

        elif normalized in databases:
            database_list.append(skill)

        elif normalized in cloud_platforms:
            cloud.append(skill)

        elif normalized in devops_tools:
            devops.append(skill)

        elif normalized in operating_systems:
            operating_system.append(skill)

        elif normalized in testing_tools:
            testing.append(skill)

        else:
            technical.append(skill)

    return {
        "technical_skills": unique_sorted(technical),
        "programming_languages": unique_sorted(programming),
        "frameworks": unique_sorted(framework_list),
        "libraries": unique_sorted(library_list),
        "databases": unique_sorted(database_list),
        "cloud_platforms": unique_sorted(cloud),
        "devops_tools": unique_sorted(devops),
        "operating_systems": unique_sorted(operating_system),
        "testing_tools": unique_sorted(testing),
    }


def make_certifications(value: Any) -> list[dict[str, Any]]:

    return [
        {
            "certification_name": name
        }
        for name in split_values(value)
    ]


def make_projects(row: pd.Series) -> list[dict[str, Any]]:

    value = row.get("Projects Count")

    try:
        count = int(value)
    except (ValueError, TypeError):
        return []

    return [
        {
            "project_name": f"Project {i}",
            "description": None,
            "technologies": []
        }
        for i in range(1, count + 1)
    ]


def make_work_experience(row: pd.Series) -> list[dict[str, Any]]:

    position = clean_value(row.get("Job Role"))

    try:
        years = float(row.get("Experience (Years)", 0))
    except (ValueError, TypeError):
        years = 0

    if not position:
        return []

    return [
        {
            "position": position,
            "company": None,
            "start_date": None,
            "end_date": None,
            "description": None,
            "years": years
        }
    ]


def convert_row(row: pd.Series) -> dict[str, Any]:

    name = clean_value(row.get("Name"))
    job_role = clean_value(row.get("Job Role"))
    education = clean_value(row.get("Education"))

    try:
        years = float(row.get("Experience (Years)", 0))
    except (ValueError, TypeError):
        years = 0

    if years.is_integer():
        years = int(years)

    skills = split_values(row.get("Skills"))

    skill_groups = classify_skills(skills)

    certifications = make_certifications(
        row.get("Certifications")
    )

    projects = make_projects(row)

    work_experience = make_work_experience(row)

    certification_names = [
        item["certification_name"]
        for item in certifications
    ]

    keywords = unique_sorted(
        skill_groups["technical_skills"]
        + skill_groups["programming_languages"]
        + skill_groups["frameworks"]
        + skill_groups["libraries"]
        + skill_groups["databases"]
        + ([job_role] if job_role else [])
        + certification_names
    )

    # Return flat structure suitable for CSV
    return {

        "candidate_name": name,

        "current_position": job_role,

        "years_of_experience": years,

        "education": education,

        "gpa": None,

        "technical_skills":
            ", ".join(skill_groups["technical_skills"]),

        "soft_skills": "",

        "programming_languages":
            ", ".join(skill_groups["programming_languages"]),

        "frameworks":
            ", ".join(skill_groups["frameworks"]),

        "libraries":
            ", ".join(skill_groups["libraries"]),

        "databases":
            ", ".join(skill_groups["databases"]),

        "cloud_platforms":
            ", ".join(skill_groups["cloud_platforms"]),

        "devops_tools":
            ", ".join(skill_groups["devops_tools"]),

        "operating_systems":
            ", ".join(skill_groups["operating_systems"]),

        "testing_tools":
            ", ".join(skill_groups["testing_tools"]),

        "certifications":
            ", ".join(certification_names),

        "projects":
            json.dumps(
                projects,
                ensure_ascii=False
            ),

        "work_experience":
            json.dumps(
                work_experience,
                ensure_ascii=False
            ),

        "achievements": "",

        "industries": "",

        "domains": "",

        "spoken_languages": "",

        "keywords":
            ", ".join(keywords),
    }


def main():

    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Found {len(df)} rows")

    results = []

    for _, row in df.iterrows():

        parsed = convert_row(row)

        results.append(parsed)

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("Conversion completed!")
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows  : {len(output_df)}")


if __name__ == "__main__":
    main()