import os
from typing import List, Set
import re

def extract_skills_from_text (text: str, skill_dict: dict) -> List[str]:
    if not isinstance(text, str):
        return []

    text_lower = text.lower()
    extracted_skils: Set[str] = set()

    for canonical_skill, synonyms in skill_dict.items():
        for synonym in synonyms:
            pattern = r'(?<!\w)' + re.escape(synonym) + r'(?!\w)'

            if re.search(pattern, text_lower):
                extracted_skils.add(canonical_skill)
                break

    return list(extracted_skils)