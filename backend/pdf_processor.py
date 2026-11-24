
from pathlib import Path
import pymupdf


def extract_text_from_pdf(pdf_path):

    extracted_pages = []

    try:
        with pymupdf.open(pdf_path) as document:

            for page_number, page in enumerate(document, start=1):

                text = page.get_text("text").strip()

                if not text:
                    print(f"Warning: No extractable text on page {page_number}")

                extracted_pages.append({
                    "text": text,
                    "page_number": page_number,
                    "source": pdf_path.name
                })

        return extracted_pages

    except Exception as error:
        print(f"Error processing {pdf_path.name}: {error}")
        return []


def process_multiple_pdfs(folder_path):

    folder = Path(folder_path)

    if not folder.exists():
        print("Error: Upload folder does not exist.")
        return []

    all_pages = []

    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return []

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        pages = extract_text_from_pdf(pdf_path)

        all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":

    pages = process_multiple_pdfs("data/uploads")

    print("\nTotal pages extracted:", len(pages))

    for page in pages[:2]:

        print("\nSource:", page["source"])
        print("Page:", page["page_number"])
        print("Text:", page["text"][:300])
