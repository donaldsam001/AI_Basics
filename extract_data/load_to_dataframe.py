import os
import pandas as pd
from .extract_doc import extract_text_from_docx
from .extract_pdf import extract_text_from_pdf


def load_cvs_to_dataframe(folder_path: str) -> pd.DataFrame:
    cv_data= []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        ext = os.path.splitext(file_name)[1].lower()

        extracted_text = ""
        if ext == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif ext == ".docx": 
            extracted_text = extract_text_from_docx(file_path)
        else:
            continue # skip other format

        cv_data.append({
            "candidate_id": os.path.splitext(file_name)[0],
            "file_name": file_name,
            "file_type": ext.replace(".", ""),
            "raw_text": extracted_text
        })

    df = pd.DataFrame(cv_data)
    return df
