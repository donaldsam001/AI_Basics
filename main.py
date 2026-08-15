import os
import pandas as pd
from extract_data.load_to_dataframe import load_cvs_to_dataframe
from text_preprocessing.text_preprocessing import clean_cv_text
from resume_parser import parse_resume



def main():
    """Load CVs, preprocess their text, and save the cleaned data."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, "example_data/docx")
    output_path = os.path.join(base_dir, "preprocessed_cvs.csv")

    # 1. Extract text from PDF and DOCX CVs.
    cvs_df = load_cvs_to_dataframe(folder_path)

    if cvs_df.empty:
        print(f"No PDF or DOCX files found in: {folder_path}")
        return

    # 2. Clean and normalize the extracted text.
    cvs_df["cleaned_text"] = cvs_df["raw_text"].apply(clean_cv_text)

    parsed_features_df = cvs_df["raw_text"].apply(parse_resume).apply(pd.Series)
    cvs_df = cvs_df.join(parsed_features_df)

    # 4. Save the raw and cleaned text for later analysis or modeling.
    cvs_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Preprocessed {len(cvs_df)} CV(s).")
    print(f"Saved cleaned data to: {output_path}")

if __name__ == "__main__":
    main()


'''
python -m src.main --cv example/Train_Resume_Data_parsed.csv --job example_data/job_roles_IT_filtered.csv
python -m src.main --cv preprocessed_cvs.csv --job example_data/jd/job_roles_IT_filtered.csv
/home/donaldsam/Downloads/AI_Basic/preprocessed_cvs.csv
/home/donaldsam/Downloads/AI_Basic/example_data/job_roles_IT_filtered.csv
'''