import fitz

def extract_pdf_text(file_path):

    doc = fitz.open(file_path)

    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text
def summarize_report(report_text):

    paragraphs = report_text.split("\n")

    cleaned = []

    for p in paragraphs:

        p = p.strip()

        if len(p) > 50:
            cleaned.append(p)

    return cleaned[:10]