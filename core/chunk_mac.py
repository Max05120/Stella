"""
chunk_mac.py

Chunks macOS technical documentation while preserving
source-authority metadata.
"""

import json
from pathlib import Path

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


INPUT_FILE = Path(
    "data/mac_processed/documents.json"
)

OUTPUT_FILE = Path(
    "data/mac_processed/chunks.json"
)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 140


def chunk_documents(
    documents: list[dict],
) -> list[dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    chunk_id = 0

    for doc in documents:

        pieces = splitter.split_text(
            doc["text"]
        )

        for piece in pieces:

            chunks.append({
                "chunk_id": chunk_id,

                "source":
                    doc["source"],

                "page":
                    doc.get("page", 1),

                "source_url":
                    doc.get("source_url"),

                "publisher":
                    doc.get("publisher"),

                "topic":
                    doc.get(
                        "topic",
                        "macos",
                    ),

                "authority":
                    doc.get(
                        "authority",
                        "unknown",
                    ),

                "authority_score":
                    doc.get(
                        "authority_score",
                        0.5,
                    ),

                "knowledge_type":
                    "mac_knowledge",

                "text": piece,
            })

            chunk_id += 1

    return chunks


if __name__ == "__main__":

    if not INPUT_FILE.exists():
        raise SystemExit(
            f"{INPUT_FILE} not found -- "
            f"run ingest_mac.py first."
        )

    documents = json.loads(
        INPUT_FILE.read_text(
            encoding="utf-8"
        )
    )

    chunks = chunk_documents(
        documents
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            chunks,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Created {len(chunks)} "
        f"Mac knowledge chunks."
    )

    print(
        f"Saved to {OUTPUT_FILE}"
    )