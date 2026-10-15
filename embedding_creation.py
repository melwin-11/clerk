import os
from typing import Any, Dict, List, Optional
import torch
from sentence_transformers import SentenceTransformer

from chunking import split_sentences
from config import (
    embedding_max_seq_length,
    embedding_model_name,
    min_chunk_tokens,
    pdf_path,
    sqlite_db_path,
)
from database import is_document_processed, save_chunks_and_embeddings
from extract_text import open_and_read_pdf


def get_embedding_model(
    model_name: str = embedding_model_name,
    device: Optional[str] = None,
    max_seq_length: int = embedding_max_seq_length,
) -> SentenceTransformer:
    """Loads and configures the sentence transformer embedding model with truncation limits."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading embedding model '{model_name}' on device '{device}'...")
    model = SentenceTransformer(model_name_or_path=model_name, device=device)
    model.max_seq_length = max_seq_length
    return model


def truncate_chunks_if_exceeds(
    chunks: List[Dict[str, Any]],
    model: SentenceTransformer,
    max_length: int = embedding_max_seq_length,
) -> List[Dict[str, Any]]:
    """
    Checks token counts using the embedding model's tokenizer and explicitly truncates
    any chunks that exceed the model's maximum sequence length.
    """
    tokenizer = model.tokenizer
    truncated_count = 0

    for item in chunks:
        token_ids = tokenizer.encode(
            item["sentence_chunk"],
            truncation=False,
            add_special_tokens=False,
        )
        if len(token_ids) > max_length:
            truncated_count += 1
            # Truncate to maximum allowed tokens
            truncated_token_ids = token_ids[:max_length]
            item["sentence_chunk"] = tokenizer.decode(
                truncated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            item["chunk_char_count"] = len(item["sentence_chunk"])
            item["chunk_word_count"] = len(item["sentence_chunk"].split())
            item["chunk_token_count"] = len(truncated_token_ids)

    if truncated_count > 0:
        print(f"[INFO] Truncated {truncated_count} chunk(s) exceeding {max_length} tokens.")

    return chunks


def process_and_embed_document(
    target_pdf: str = pdf_path,
    db_path: str = sqlite_db_path,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extracts text, splits into sentence chunks, enforces token bounds,
    generates embeddings with all-mpnet-base-v2, and stores everything in SQLite.
    """
    if not os.path.exists(target_pdf):
        raise FileNotFoundError(f"PDF file not found at: {target_pdf}")

    if not force and is_document_processed(file_path=target_pdf, db_path=db_path):
        print(f"[INFO] Document '{target_pdf}' is already embedded and up-to-date in SQLite.")
        return []

    print(f"[INFO] Extracting and chunking PDF: '{target_pdf}'...")
    raw_pages = open_and_read_pdf(pdf_path=target_pdf)
    all_chunks = split_sentences(pages_and_text=raw_pages)

    filtered_chunks = [
        item for item in all_chunks if item["chunk_token_count"] >= min_chunk_tokens
    ]

    if not filtered_chunks:
        print(f"[WARNING] No chunks found with token counts >= {min_chunk_tokens}.")
        return []

    embedding_model = get_embedding_model()

    # Safety truncation for all-mpnet sequence limit
    filtered_chunks = truncate_chunks_if_exceeds(
        chunks=filtered_chunks,
        model=embedding_model,
        max_length=embedding_max_seq_length,
    )

    print(f"[INFO] Encoding {len(filtered_chunks)} chunks into vector embeddings...")
    texts_to_embed = [item["sentence_chunk"] for item in filtered_chunks]

    embeddings = embedding_model.encode(
        texts_to_embed,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True,
    )

    for item, emb in zip(filtered_chunks, embeddings):
        item["embedding"] = emb.cpu().numpy()

    print(f"[INFO] Storing chunks and embeddings into SQLite at '{db_path}'...")
    saved_count = save_chunks_and_embeddings(
        chunks=filtered_chunks,
        file_path=target_pdf,
        db_path=db_path,
    )

    print(f"[SUCCESS] Successfully saved {saved_count} chunks to SQLite.")
    return filtered_chunks


def main():
    return process_and_embed_document()


if __name__ == "__main__":
    main()
