"""
ingest.py
Loads raw documents (.pdf, .docx, .txt) from data/raw/ and converts
them into a single list of "documents" -- plain text plus metadata --
ready for chunking in the next step.
"""
import json
from pathlib import Path
from pypdf import PdfReader
import docx

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/documents.json")

"""Extract text from a PDF, one record per page."""
def load_pdf(file_path: Path) -> list[dict]:
    records = []
    reader = PdfReader(str(file_path))
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            records.append({
                "source": file_path.name,
                "page": page_num,
                "text": text,
            })
    return records

"""Extract text from a Word document as a single record."""
def loaddocx(file_path: Path) -> list[dict]:
    doc = docx.Document(str(file_path))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not full_text.strip():
        return []
    return [{"source": file_path.name,
            "page": 1,
            "text": full_text}]
    
"""Load a plain text file as a single record."""
def loadtxt(file_path: Path) -> list[dict]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []
    return [{"source": file_path.name,
             "page": 1,
             "text": text}]

LOADERS = {".pdf": load_pdf,".docx": loaddocx,".txt": loadtxt}

def load_all_documents(raw_dir: Path = RAW_DIR) -> list[dict]:
    all_records = []
    files = sorted(raw_dir.glob("*"))
    if not files:
        print(f"No files found in {raw_dir}/ -- add some PDFs, DOCX, or TXT files first.")
        return all_records
    
    for file_path in files:
        loader = LOADERS.get(file_path.suffix.lower())
        if loader is None:
            print(f"Skipping unsupported file type: {file_path.name}")
            continue
        try:
            records = loader(file_path)
            all_records.extend(records)
            print(f"Loaded {file_path.name}: {len(records)} record(s)")
        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")
            
    return all_records

if __name__ == "__main__":
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    documents = load_all_documents()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(documents)} document record(s) to {OUTPUT_FILE}")



