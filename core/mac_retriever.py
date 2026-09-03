"""
mac_retriever.py

Retrieval layer for Stella's dedicated
macOS technical knowledge base.
"""

import chromadb

from core.embed import (
    embed_text,
    CHROMA_DIR,
)


COLLECTION_NAME = "mac_knowledge"


def get_collection():

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_collection(
        COLLECTION_NAME
    )


def retrieve_mac(
    query: str,
    top_k: int = 6,
) -> list[dict]:

    collection = get_collection()

    query_embedding = embed_text(
        query
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],

        n_results=top_k,

        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (
        results["documents"][0]
    )

    metadatas = (
        results["metadatas"][0]
    )

    distances = (
        results["distances"][0]
    )

    retrieved = []

    for (
        document,
        metadata,
        distance,
    ) in zip(
        documents,
        metadatas,
        distances,
    ):

        retrieved.append({
            "text":
                document,

            "source":
                metadata.get(
                    "source"
                ),

            "source_url":
                metadata.get(
                    "source_url"
                ),

            "topic":
                metadata.get(
                    "topic"
                ),

            "publisher":
                metadata.get(
                    "publisher"
                ),

            "authority":
                metadata.get(
                    "authority"
                ),

            "authority_score":
                metadata.get(
                    "authority_score",
                    0.5,
                ),

            "distance":
                distance,
        })

    # Prefer semantically relevant
    # authoritative documentation.
    retrieved.sort(
        key=lambda r: (
            r["distance"],
            -float(
                r["authority_score"]
            ),
        )
    )

    return retrieved


if __name__ == "__main__":

    query = input(
        "Mac question: "
    )

    results = retrieve_mac(
        query
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- Result "
            f"{index} ---"
        )

        print(
            f"Topic: "
            f"{result['topic']}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"Authority: "
            f"{result['authority']} "
            f"({result['authority_score']})"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print()
        print(
            result["text"]
        )