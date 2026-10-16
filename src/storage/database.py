import hashlib
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

try:
    from src.config import sqlite_db_path
except ImportError:
    from config import sqlite_db_path


def compute_file_hash(file_path: str) -> str:
    """Computes SHA-256 hash for a given file to detect changes."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_connection(db_path: str = sqlite_db_path) -> sqlite3.Connection:
    """Creates a connection to SQLite database with WAL mode enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: str = sqlite_db_path) -> None:
    """Initializes tables for document metadata and chunk embeddings."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS document_meta (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                sentence_chunk TEXT NOT NULL,
                chunk_char_count INTEGER NOT NULL,
                chunk_word_count INTEGER NOT NULL,
                chunk_token_count REAL NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (file_path) REFERENCES document_meta (file_path)
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON chunks(file_hash);"
        )
        conn.commit()


def is_document_processed(file_path: str, db_path: str = sqlite_db_path) -> bool:
    """Checks whether the document exists and its SHA-256 hash matches the stored hash."""
    if not os.path.exists(file_path):
        return False

    current_hash = compute_file_hash(file_path)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_hash, chunk_count FROM document_meta WHERE file_path = ?",
            (file_path,),
        )
        row = cursor.fetchone()
        if not row:
            return False

        stored_hash, chunk_count = row
        if stored_hash != current_hash or chunk_count == 0:
            return False

        cursor.execute(
            "SELECT COUNT(1) FROM chunks WHERE file_path = ?", (file_path,)
        )
        actual_chunks = cursor.fetchone()[0]
        return actual_chunks == chunk_count


def save_chunks_and_embeddings(
    chunks: List[Dict[str, Any]],
    file_path: str,
    db_path: str = sqlite_db_path,
) -> int:
    """
    Saves chunks and embeddings into SQLite.
    Overwrites previous entries for the same file_path if it was modified.
    """
    init_db(db_path)
    file_hash = compute_file_hash(file_path)
    now_iso = datetime.utcnow().isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        cursor.execute("DELETE FROM document_meta WHERE file_path = ?", (file_path,))

        cursor.execute(
            """
            INSERT INTO document_meta (file_path, file_hash, chunk_count, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (file_path, file_hash, len(chunks), now_iso),
        )

        rows = []
        for chunk in chunks:
            raw_embedding = chunk["embedding"]
            if isinstance(raw_embedding, torch.Tensor):
                emb_array = raw_embedding.detach().cpu().numpy().astype(np.float32)
            elif isinstance(raw_embedding, np.ndarray):
                emb_array = raw_embedding.astype(np.float32)
            else:
                emb_array = np.array(raw_embedding, dtype=np.float32)

            rows.append(
                (
                    file_path,
                    file_hash,
                    int(chunk["page_number"]),
                    str(chunk["sentence_chunk"]),
                    int(chunk["chunk_char_count"]),
                    int(chunk["chunk_word_count"]),
                    float(chunk["chunk_token_count"]),
                    emb_array.tobytes(),
                )
            )

        cursor.executemany(
            """
            INSERT INTO chunks (
                file_path, file_hash, page_number, sentence_chunk,
                chunk_char_count, chunk_word_count, chunk_token_count, embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    return len(chunks)


def load_all_embeddings(
    db_path: str = sqlite_db_path,
) -> Tuple[Optional[torch.Tensor], List[Dict[str, Any]]]:
    """
    Loads all chunk embeddings as a PyTorch tensor and metadata records as dictionaries.
    Returns:
        (embeddings_tensor, chunk_records)
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, page_number, sentence_chunk, chunk_char_count,
                   chunk_word_count, chunk_token_count, embedding
            FROM chunks
            ORDER BY id ASC
            """
        )
        rows = cursor.fetchall()

    if not rows:
        return None, []

    chunk_records: List[Dict[str, Any]] = []
    embeddings_list: List[np.ndarray] = []

    for row in rows:
        (
            chunk_id,
            page_number,
            sentence_chunk,
            chunk_char_count,
            chunk_word_count,
            chunk_token_count,
            emb_blob,
        ) = row

        emb_array = np.frombuffer(emb_blob, dtype=np.float32)
        embeddings_list.append(emb_array)

        chunk_records.append(
            {
                "id": chunk_id,
                "page_number": page_number,
                "sentence_chunk": sentence_chunk,
                "chunk_char_count": chunk_char_count,
                "chunk_word_count": chunk_word_count,
                "chunk_token_count": chunk_token_count,
            }
        )

    all_embeddings_np = np.vstack(embeddings_list)
    embeddings_tensor = torch.from_numpy(all_embeddings_np).to(torch.float32)

    return embeddings_tensor, chunk_records


def get_db_stats(db_path: str = sqlite_db_path) -> Dict[str, Any]:
    """Returns basic counts and stats of the SQLite vector database."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(1) FROM document_meta")
        doc_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(1) FROM chunks")
        chunk_count = cursor.fetchone()[0]

        cursor.execute("SELECT file_path, chunk_count, updated_at FROM document_meta")
        documents = [
            {"file_path": r[0], "chunks": r[1], "updated_at": r[2]}
            for r in cursor.fetchall()
        ]

    return {
        "db_path": db_path,
        "total_documents": doc_count,
        "total_chunks": chunk_count,
        "documents": documents,
    }


if __name__ == "__main__":
    print("[INFO] Checking SQLite database status...")
    stats = get_db_stats()
    print(f"[INFO] Database: {stats['db_path']}")
    print(f"[INFO] Documents tracked: {stats['total_documents']}")
    print(f"[INFO] Total Chunks: {stats['total_chunks']}")
    for doc in stats["documents"]:
        print(f"  - {doc['file_path']} ({doc['chunks']} chunks, updated {doc['updated_at']})")
