"""
embed.py
Reads data/processed/chunks.json, generates an embedding for each
chunk using Ollama's nomic-embed-text model, and stores everything
in a local, persistent ChromaDB collection.
"""

import json
from pathlib import Path
import chromadb
import ollama

CHUNK_FILE = Path("data/processed/chunks.json")
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "stella"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 20  # Number of chunks to embed at once

def embed_text(text: str) -> list[float]:
    """Generate an embedding for a single text chunk using Ollama."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response['embedding']

def main():
    if not CHUNK_FILE.exists():
        raise SystemExit(f"{CHUNK_FILE} not found -- run chunk.py first.")

    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        ids = [str(c["chunk_id"]) for c in batch]
        documents = [c["text"] for c in batch]
        embeddings = [embed_text(c['text']) for c in batch]
        metadatas = [{"source": c["source"], "page": c["page"]} for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"Embedded and upserted chunks {start + 1} to {min(start + BATCH_SIZE, total)} of {total}.")

    print(f"Embedded {len(chunks)} chunks and saved to ChromaDB collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()
