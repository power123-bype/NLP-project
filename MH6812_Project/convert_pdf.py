import pdfplumber
import os

pdf_path = "task/nlp分工.pdf"
txt_path = "task/nlp_task_assignment.txt"

if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
    exit(1)

try:
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"--- Page {i+1} ---\n{page_text}\n\n"
            else:
                text += f"--- Page {i+1} ---\n[No text extracted]\n\n"
            
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Successfully extracted text to {txt_path}")
except Exception as e:
    print(f"Error: {e}")
