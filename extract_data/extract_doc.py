import docx

def extract_text_from_docx(file_path: str) -> str:
    doc= docx.Document(file_path)
    full_text = []

    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())

    return "\n".join(full_text)


# file_path="/home/donaldsam/Downloads/AI_Basic/example_data/Professional_Software_Engineer_CV_Template.docx"
# result=extract_text_from_docx(file_path)

# print(result)