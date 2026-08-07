import os
from typing import List, Set
import re

def extract_items(text: str, skill_dict: dict) -> List[str]:
    if not isinstance(text, str):
        return []

    text_lower = text.lower()
    result: Set[str] = set()

    for canonical_skill, synonyms in skill_dict.items():
        for synonym in synonyms:
            pattern = r'(?<!\w)' + re.escape(synonym) + r'(?!\w)'

            if re.search(pattern, text_lower):
                result.add(canonical_skill)
                break

    return sorted(result)

def extract_experience(text: str):
    pattern =  r'(\d+)\+?\s*(?:years?|yrs?)'
    match = re.search(pattern, text.lower())

    if match: 
        return int(match.group(1))

    return None

def extract_email(text: str):
    match = re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',text)

    return match.group(0) if match else None

def extract_phone(text: str):
    match = re.search( r'(\+?\d[\d\s().-]{8,}\d)',text)

    return match.group(0) if match else None