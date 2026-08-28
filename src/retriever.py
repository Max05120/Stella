"""
retriever.py

Retrieves semantically relevant chunks from ChromaDB.
"""

import chromadb

from .embed import (
    embed_text,
    CHROMA_DIR,
    COLLECTION_NAME,
)


def get_collection():
    """Connect to the persistent ChromaDB collection."""

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    return client.get_collection(
        COLLECTION_NAME
    )


def retrieve(
    query: str,
    top_k: int = 5
) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.
    """

    collection = get_collection()

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved.append(
            {
                "text": document,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "distance": distance,
            }
        )

    return retrieved


if __name__ == "__main__":

    query = input("Question: ")

    results = retrieve(query)

    for i, result in enumerate(results, start=1):

        print(
            f"\n--- Result {i} ---"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Page: {result['page']}"
        )

        print(
            f"Distance: {result['distance']:.4f}"
        )

        print(
            result["text"]
        )

