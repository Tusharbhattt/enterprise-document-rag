
from pathlib import Path

import chromadb

from parser import extract_pdf_pages
from chunker import create_semantic_chunks, generate_embeddings, embedding_model


# Project root: enterprise-document-rag
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Store ChromaDB data persistently inside the project
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_or_create_collection(
    name="document_chunks",
    metadata={"hnsw:space": "cosine"},
)


def index_chunks_to_db(chunks):
    """Store embedded text chunks in ChromaDB."""

    if not chunks:
        print("No chunks available to index.")
        return

    ids = [
        f"chunk_{chunk['chunk_id']}"
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = [
        chunk["embedding"]
        for chunk in chunks
    ]

    metadatas = [
        {"page_number": chunk["page_number"]}
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Successfully indexed {len(chunks)} chunks.")


def query_vector_db(query_text, n_results=3):
    """Find the most relevant chunks for a query."""

    query_embedding = embedding_model.encode(
        [query_text]
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    retrieved_chunks = []

    if not results["ids"] or not results["ids"][0]:
        print("No matching chunks found.")
        return retrieved_chunks

    for i in range(len(results["ids"][0])):
        retrieved_chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "page_number": results["metadatas"][0][i]["page_number"],
            "distance": results["distances"][0][i],
        })

    return retrieved_chunks


if __name__ == "__main__":

    # 1. Locate the sample PDF
    pdf_path = PROJECT_ROOT / "data" / "sample.pdf"

    # 2. Extract text from PDF pages
    pages_data = extract_pdf_pages(str(pdf_path))

    # 3. Split extracted text into chunks
    chunks = create_semantic_chunks(pages_data)

    print(f"Created {len(chunks)} chunks.")

    # 4. Generate embeddings for chunks
    embedded_chunks = generate_embeddings(chunks)

    # 5. Store chunks and embeddings in ChromaDB
    index_chunks_to_db(embedded_chunks)

    # 6. Search the database
    question = "What is this document about?"

    retrieved_chunks = query_vector_db(
        question,
        n_results=2,
    )

    # 7. Display search results
    print("\n--- Retrieved Results ---")

    for result in retrieved_chunks:
        print(f"\nChunk ID: {result['id']}")
        print(f"Page: {result['page_number']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Text: {result['text'][:300]}")
