import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import discord_channel_id, discord_token, pdf_path, sqlite_db_path
from src.ingestion.embedding_creation import process_and_embed_document
from src.storage.database import get_db_stats, is_document_processed
from src.rag.rag import RAGEngine
from src.bot.bot import ClerkDiscordBot


def run_pipeline(target_pdf: str = pdf_path, db_path: str = sqlite_db_path) -> RAGEngine:
    """
    Executes the automated RAG pipeline:
    1. Checks if the PDF exists and whether its SHA-256 hash matches the SQLite record.
    2. If new or changed, extracts, chunks, truncates, embeds, and updates SQLite.
    3. Loads embeddings into the RAG engine.
    """
    print("=" * 60)
    print("Clerk Local-RAG Systematic Pipeline")
    print("=" * 60)

    if not os.path.exists(target_pdf):
        raise FileNotFoundError(f"[ERROR] Target document not found at: '{target_pdf}'")

    print(f"[PIPELINE] Checking document status for: '{target_pdf}'")

    if not is_document_processed(file_path=target_pdf, db_path=db_path):
        print(f"[PIPELINE] PDF is new or modified. Running ingestion & embedding pipeline...")
        chunks = process_and_embed_document(target_pdf=target_pdf, db_path=db_path, force=True)
        print(f"[PIPELINE] Embedded and stored {len(chunks)} chunks into SQLite.")
    else:
        print(f"[PIPELINE] Document is up-to-date in SQLite. Skipping re-embedding.")

    stats = get_db_stats(db_path=db_path)
    print(f"[PIPELINE] Database ready: {stats['total_chunks']} total chunks tracked across {stats['total_documents']} document(s).")

    print(f"[PIPELINE] Initializing RAG Engine...")
    engine = RAGEngine(db_path=db_path)
    return engine


def main():
    """Main systematic loop: validates PDF/DB pipeline then boots Discord Bot."""
    engine = run_pipeline()

    if not discord_token:
        print("[WARNING] DISCORD_TOKEN is not configured in .env. Bot serving skipped.")
        print("[INFO] You can test queries directly in terminal using: uv run python src/rag/rag.py")
        return

    print(f"[PIPELINE] Starting Clerk Discord Bot...")
    bot = ClerkDiscordBot(
        rag_engine=engine,
        target_channel_id=discord_channel_id,
    )
    bot.run(discord_token)


if __name__ == "__main__":
    main()
