
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import pandas as pd

from chunking import split_sentences
from config import pdf_path
from extract_text import open_and_read_pdf


def main():
    print("Extracting and chunking PDF text...")

    all_chunks = split_sentences(
        pages_and_text=open_and_read_pdf(pdf_path=pdf_path)
    )

    filtered_chunks = [
        item for item in all_chunks
        if item['chunk_token_count'] >= 71
    ]

    if not filtered_chunks:
        print("No chunks found with token counts >= 71.")
        return

    print(f"Loading model on CUDA. Processing {len(filtered_chunks)} chunks...")

    embedding_model = SentenceTransformer(
        model_name_or_path='all-mpnet-base-v2',
        device='cuda'
    )

    texts_to_embed = [
        item['sentence_chunk'] for item in filtered_chunks
    ]

    embeddings = embedding_model.encode(
        texts_to_embed,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True
    )

    for item, embedding in zip(filtered_chunks, embeddings):
        item['embedding'] = embedding.cpu().numpy()

    # Save embeddings to a CSV file
    print("Saving embeddings to file...")

    text_chunks_and_embeddings_df = pd.DataFrame(filtered_chunks)

    embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"

    text_chunks_and_embeddings_df.to_csv(
        embeddings_df_save_path,
        index=False
    )

    print(f"Embeddings saved to: {embeddings_df_save_path}")

    return filtered_chunks


if __name__ == "__main__":
    processed_chunks = main()
