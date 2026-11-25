
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from parser import extract_pdf_pages


# Load the embedding model.
# This model generates 384-dimensional embeddings.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def create_semantic_chunks(
    pages_data,
    chunk_size=500,
    chunk_overlap=50,
):
    """
    Split extracted PDF page text into overlapping chunks.
    Preserve the original page number for every chunk.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    chunk_id = 1

    for page in pages_data:
        page_number = page["page_number"]
        page_text = page["content"]

        if not page_text.strip():
            continue

        page_chunks = text_splitter.split_text(page_text)

        for chunk_text in page_chunks:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "text": chunk_text,
                }
            )

            chunk_id += 1

    return chunks


def generate_embeddings(chunks):
    """
    Generate a 384-dimensional vector for each chunk.
    """

    if not chunks:
        return chunks

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
    )

    for index, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[index].tolist()

    return chunks


if __name__ == "__main__":
    sample_pdf = "data/sample.pdf"

    try:
        # Step 1: Extract PDF pages using Day 1 parser
        pages = extract_pdf_pages(sample_pdf)

        # Step 2: Create chunks
        chunks = create_semantic_chunks(pages)

        print(
            f"\nCreated {len(chunks)} chunks "
            f"from {len(pages)} PDF pages."
        )

        # Step 3: Generate embeddings
        embedded_chunks = generate_embeddings(chunks)

        print(
            f"Generated embeddings for "
            f"{len(embedded_chunks)} chunks."
        )

        # Step 4: Inspect the first chunk
        if embedded_chunks:
            first_chunk = embedded_chunks[0]

            print("\n--- First Chunk ---")
            print(f"Chunk ID: {first_chunk['chunk_id']}")
            print(f"Page Number: {first_chunk['page_number']}")
            print(f"Text: {first_chunk['text']}")

            print(
                "Embedding dimensions: "
                f"{len(first_chunk['embedding'])}"
            )

            print(
                "First 5 vector values: "
                f"{first_chunk['embedding'][:5]}"
            )

    except Exception as error:
        print(f"Error: {error}")
