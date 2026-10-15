from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from chunking import split_sentences
from config import pdf_path
from extract_text import open_and_read_pdf

def main():
    # 1. Fetch chunks and filter for token count >= 71
    print("Extracting and chunking PDF text...")
    all_chunks = split_sentences(pages_and_text=open_and_read_pdf(pdf_path=pdf_path))
    
    # Filter chunks natively using list comprehension
    filtered_chunks = [item for item in all_chunks if item['chunk_token_count'] >= 71]
    
    if not filtered_chunks:
        print("No chunks found with token counts >= 71.")
        return

    # 2. Initialize the embedding model on CUDA directly
    print(f"Loading model on CUDA. Processing {len(filtered_chunks)} chunks...")
    embedding_model = SentenceTransformer(
        model_name_or_path='all-mpnet-base-v2', # Shortened identifier
        device='cuda'
    )

    # 3. Batch encode for massive speedup on GPU
    # Extract just the raw strings into a list
    texts_to_embed = [item['sentence_chunk'] for item in filtered_chunks]
    
    # Encode all at once using internal batching (automatically utilizes GPU parallelism)
    embeddings = embedding_model.encode(
        texts_to_embed, 
        batch_size=32, 
        show_progress_bar=True
    )

    # 4. Map the embeddings back to your dictionary items
    for item, embedding in zip(filtered_chunks, embeddings):
        item['embedding'] = embedding

    print("Embedding complete!")
    return filtered_chunks

if __name__ == "__main__":
    processed_chunks = main()
