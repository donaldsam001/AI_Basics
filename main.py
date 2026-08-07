import os
from extract_data.load_to_dataframe import load_cvs_to_dataframe
from text_preprocessing.text_preprocessing import clean_cv_text
from text_preprocessing.extract_skills import extract_items, extract_email, extract_experience, extract_phone
import json
from pathlib import Path



def main():
    """Load CVs, preprocess their text, and save the cleaned data."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, "example_data")
    output_path = os.path.join(base_dir, "preprocessed_cvs.csv")

    # 1. Extract text from PDF and DOCX CVs.
    cvs_df = load_cvs_to_dataframe(folder_path)

    if cvs_df.empty:
        print(f"No PDF or DOCX files found in: {folder_path}")
        return

    # 2. Clean and normalize the extracted text.
    cvs_df["cleaned_text"] = cvs_df["raw_text"].apply(clean_cv_text)

    # 3. Extract skills
    skill_file = Path(__file__).parent / "dictionary" / "skill_dictionary.json"

    with skill_file.open(encoding="utf-8") as file:
        skill_dictionary = json.load(file)
        cvs_df['extracted_skills'] = cvs_df["cleaned_text"].apply(lambda x: extract_items(x, skill_dictionary))


    # 4. Save the raw and cleaned text for later analysis or modeling.
    cvs_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Preprocessed {len(cvs_df)} CV(s).")
    print(f"Saved cleaned data to: {output_path}")

if __name__ == "__main__":
    main()
