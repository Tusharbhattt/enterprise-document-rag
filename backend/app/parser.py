
from pypdf import PdfReader
from pathlib import Path


def extract_pdf_pages(pdf_path: str):
    """Extract text page by page from a PDF."""

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    reader = PdfReader(str(pdf_file))

    extracted_pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = text.strip()

        if text:
            extracted_pages.append({
                "page_number": page_num,
                "content": text
            })

    return extracted_pages


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]
    sample_pdf = project_root / "data" / "sample.pdf"

    try:
        pages = extract_pdf_pages(sample_pdf)

        print(f"Successfully read {len(pages)} pages!\n")

        for page in pages:
            print(f"--- Page {page['page_number']} ---")
            print(page["content"][:300])
            print()

    except Exception as e:
        print(f"Error: {e}")

