import pdfplumber

def extract_text_from_pdf(file_path)-> str:
    full_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text.strip())

    return "\n".join(full_text)

# file_path="/home/donaldsam/Downloads/AI_Basic/example_data/NongVanSam_Backend_Java.pdf"
# result= extract_text_from_pdf(file_path)
# print(result)