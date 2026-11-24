from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)

    return chunks

if __name__ == "__main__":

    sample_text = "Artificial Intelligence is transforming the world. " * 50

    chunks = split_text_into_chunks(sample_text)

    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {i} length:", len(chunk))
        print(chunk)

    for i in range(len(chunks) - 1):

     current_chunk = chunks[i]
    next_chunk = chunks[i + 1]

    print(f"\nComparing Chunk {i+1} and Chunk {i+2}")

    print("Current ending:", current_chunk[-60:])
    print("Next beginning:", next_chunk[:60])


def create_chunks_with_metadata(pages):

    all_chunks = []

    for page in pages:

        text = page["text"]

        if not text.strip():
            continue

        chunks = split_text_into_chunks(text)

        for chunk_index, chunk_text in enumerate(chunks, start=1):

            chunk_data = {
                "chunk_id": f'{Path(page["source"]).stem}_p{page["page_number"]}_chunk{chunk_index}',
                "text": chunk_text,
                "source": page["source"],
                "page_number": page["page_number"]
            }

            all_chunks.append(chunk_data)

    return all_chunks
