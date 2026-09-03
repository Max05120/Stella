"""
search.py
Sanity-check the vector store built by embed.py -- run a similarity
search and see which chunks come back for a given query.
Usage: python3 src/search.py your question here
"""
import sys
import chromadb
from core.embed import embed_text, CHROMA_DIR, COLLECTION_NAME

def search(query: str, top_k: int = 5):
    """Search the ChromaDB collection for the most similar chunks to the query."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    # Generate an embedding for the query
    query_embedding = embed_text(query)

    # Perform a similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    distances = results['distances'][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        preview = doc[:200] + "..." if len(doc) > 200 else doc
        print(f"\n#{i} (distance: {dist:.4f}) -- {meta['source']} p.{meta['page']}:\n{preview}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/search.py your question here")
        sys.exit(1)

    query = " ".join(sys.argv[1:]) or "What is this document about?"
    print(f"Searching for: {query}")
    search(query)