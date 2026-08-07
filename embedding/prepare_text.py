import re

def prepare_text_for_model (text: str)-> str:
    if text is None:
        return ""

    text = re.sub(r"\s+", " ", text)

    text = text.lower()

    return text.strip()
