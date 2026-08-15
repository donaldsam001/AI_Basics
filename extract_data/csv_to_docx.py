import pandas as pd
from docx import Document
import argparse
import os


def csv_to_docx(csv_file, output_file=None):
    # Read CSV
    df = pd.read_csv(csv_file)

    # Output filename
    if output_file is None:
        output_file = os.path.splitext(csv_file)[0] + "_formatted.docx"

    # Create Word document
    doc = Document()

    # Process each row
    for _, row in df.iterrows():

        parts = []

        for column in df.columns:
            value = row[column]

            # Skip empty values
            if pd.isna(value):
                continue

            value = str(value).strip()

            if not value:
                continue

            # Format:
            # column: content
            parts.append(f"{column}: {value}")

        # Join all columns into one line
        line = ". ".join(parts)

        # Add final period
        if line:
            line += "."

        doc.add_paragraph(line)

    # Save DOCX
    doc.save(output_file)

    print(f"Successfully created: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert every CSV row into a formatted DOCX paragraph."
    )

    parser.add_argument(
        "csv_file",
        help="Path to CSV file"
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output DOCX filename"
    )

    args = parser.parse_args()

    csv_to_docx(
        args.csv_file,
        args.output
    )