"""
embed_mac.py

Embeds dedicated macOS knowledge into the
mac_knowledge ChromaDB collection.
"""

import json
from pathlib import Path

import chromadb

from core.embed import (
    embed_text,
    CHROMA_DIR,
)


CHUNK_FILE = Path(
    "data/mac_processed/chunks.json"
)

COLLECTION_NAME = "mac_knowledge"

BATCH_SIZE = 20


def main():

    if not CHUNK_FILE.exists():
        raise SystemExit(
            f"{CHUNK_FILE} not found -- "
            f"run chunk_mac.py first."
        )

    chunks = json.loads(
        CHUNK_FILE.read_text(
            encoding="utf-8"
        )
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:
        client.delete_collection(
        COLLECTION_NAME
        )

        print(
        f"Removed existing "
        f"{COLLECTION_NAME} collection."
        )
        
    except Exception:
        pass

    collection = (
        client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description":
                    "Stella macOS technical knowledge"
            },
        )
    )

    total = len(chunks)

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        batch = chunks[
            start:start + BATCH_SIZE
        ]

        ids = [
            f"mac_{c['chunk_id']}"
            for c in batch
        ]

        documents = [
            c["text"]
            for c in batch
        ]

        embeddings = [
            embed_text(c["text"])
            for c in batch
        ]

        metadatas = []

        for c in batch:

            metadata = {
                "source":
                    c["source"],

                "page":
                    c.get(
                        "page",
                        1,
                    ),

                "publisher":
                    c.get(
                        "publisher",
                        "Unknown",
                    ),

                "topic":
                    c.get(
                        "topic",
                        "macos",
                    ),

                "authority":
                    c.get(
                        "authority",
                        "unknown",
                    ),

                "authority_score":
                    float(
                        c.get(
                            "authority_score",
                            0.5,
                        )
                    ),

                "knowledge_type":
                    "mac_knowledge",
            }

            # Chroma metadata values
            # cannot be None.
            if c.get("source_url"):
                metadata["source_url"] = (
                    c["source_url"]
                )

            metadatas.append(metadata)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(
            f"Embedded "
            f"{min(start + BATCH_SIZE, total)}"
            f"/{total}"
        )

    print()
    print(
        f"Mac knowledge ready."
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Chunks: {total}"
    )


if __name__ == "__main__":
    main()