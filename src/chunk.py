"""
chunk.py
Splits loaded documents into overlapping chunks sized for embedding.
Reads data/processed/documents.json, writes data/processed/chunks.json
"""
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = Path("data/processed/documents.json")
OUTPUT_FILE = Path("data/processed/chunks.json")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

def chunk_documents(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = []
    chunk_id = 0
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for piece in pieces:
            chunks.append({
                "chunk_id": chunk_id,
                "source": doc["source"],
                "page": doc["page"],
                "text": piece,
            })
            chunk_id += 1
    return chunks

if __name__ == "__main__":
    INPUT_FILE = Path("data/processed/documents.json")
    OUTPUT_FILE = Path("data/processed/chunks.json")

    if not INPUT_FILE.exists():
        raise SystemExit(f"{INPUT_FILE} not found -- run ingest.py first.")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    chunks = chunk_documents(documents)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Created {len(chunks)} chunk(s) from {len(documents)} document record(s)")
    print(f"Saved to {OUTPUT_FILE}")
