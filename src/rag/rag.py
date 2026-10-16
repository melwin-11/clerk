import os
from typing import Any, Dict, List, Optional, Tuple
import torch
from sentence_transformers import SentenceTransformer, util
import ollama

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from src.config import (
        embedding_max_seq_length,
        embedding_model_name,
        ollama_host,
        ollama_model,
        ollama_num_ctx,
        ollama_num_gpu,
        ollama_temperature,
        rag_embedding_device,
        sqlite_db_path,
    )
    from src.storage.database import load_all_embeddings
except ImportError:
    from config import (
        embedding_max_seq_length,
        embedding_model_name,
        ollama_host,
        ollama_model,
        ollama_num_ctx,
        ollama_num_gpu,
        ollama_temperature,
        rag_embedding_device,
        sqlite_db_path,
    )
    from database import load_all_embeddings


class RAGEngine:
    def __init__(
        self,
        db_path: str = sqlite_db_path,
        embedding_model_name: str = embedding_model_name,
        embedding_device: str = rag_embedding_device,
        ollama_host: str = ollama_host,
        ollama_model: str = ollama_model,
        num_gpu: Optional[int] = ollama_num_gpu,
        num_ctx: int = ollama_num_ctx,
        temperature: float = ollama_temperature,
    ):
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.device = embedding_device
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.num_gpu = num_gpu
        self.num_ctx = num_ctx
        self.temperature = temperature

        print(f"[INFO] Initializing RAG Embedding Model ({self.embedding_model_name}) on {self.device}...")
        self.embedding_model = SentenceTransformer(
            model_name_or_path=self.embedding_model_name,
            device=self.device,
        )
        self.embedding_model.max_seq_length = embedding_max_seq_length

        self.ollama_client = ollama.Client(host=self.ollama_host)

        # Cache embeddings from SQLite
        self.embeddings: Optional[torch.Tensor] = None
        self.chunks_meta: List[Dict[str, Any]] = []
        self.refresh_embeddings()

    def refresh_embeddings(self) -> int:
        """Loads or reloads embeddings from SQLite into memory for fast similarity search."""
        embeddings_tensor, chunks_meta = load_all_embeddings(db_path=self.db_path)
        if embeddings_tensor is not None:
            self.embeddings = embeddings_tensor.to(self.device)
            self.chunks_meta = chunks_meta
            print(f"[INFO] Loaded {len(self.chunks_meta)} chunk embeddings from SQLite into {self.device}.")
            return len(self.chunks_meta)
        else:
            self.embeddings = None
            self.chunks_meta = []
            print("[WARNING] No embeddings found in SQLite. Run document ingestion first.")
            return 0

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Embeds the query and computes dot product similarity against stored chunk embeddings.
        Returns top-k matching chunks with similarity scores.
        """
        if self.embeddings is None or len(self.chunks_meta) == 0:
            count = self.refresh_embeddings()
            if count == 0:
                return []

        # Encode query with max_seq_length truncation
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).to(self.device)

        # Compute dot scores / cosine similarity
        dot_scores = util.dot_score(query_embedding, self.embeddings)[0]

        k = min(top_k, len(self.chunks_meta))
        scores, indices = torch.topk(input=dot_scores, k=k)

        results = []
        for score, idx in zip(scores, indices):
            item = dict(self.chunks_meta[idx.item()])
            item["score"] = float(score.cpu().item())
            results.append(item)

        return results

    def format_prompt(self, query: str, context_items: List[Dict[str, Any]]) -> str:
        """Constructs an augmented context prompt for the Ollama model."""
        context_str = "\n".join(
            [
                f"[Page {item['page_number']} - Relevance: {item['score']:.4f}]\n{item['sentence_chunk']}"
                for item in context_items
            ]
        )

        prompt = f"""You are a precise, knowledgeable assistant answering questions based on the provided document context.
Based on the following extracted passages, answer the user query accurately and concisely.
If the information cannot be found in the context, say that the document does not contain that information.

Context passages:
{context_str}

User query: {query}
Answer:"""
        return prompt

    def query(
        self,
        user_query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Performs end-to-end RAG:
        1. Retrieve top-k context passages from SQLite.
        2. Format augmented prompt.
        3. Generates response using local Ollama model.
        """
        context_items = self.retrieve_relevant_chunks(query=user_query, top_k=top_k)

        if not context_items:
            return {
                "query": user_query,
                "answer": "No relevant document embeddings were found in the database. Please ensure document ingestion has completed.",
                "context_items": [],
            }

        prompt = self.format_prompt(query=user_query, context_items=context_items)

        print(f"[INFO] Generating response from Ollama model '{self.ollama_model}'...")
        options: Dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
        }
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu

        response = self.ollama_client.chat(
            model=self.ollama_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise, factual assistant. Answer strictly according to the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            options=options,
        )

        answer_text = response["message"]["content"]

        return {
            "query": user_query,
            "answer": answer_text,
            "context_items": context_items,
        }


def main():
    """Standalone CLI loop for testing RAG queries directly from the terminal."""
    print("=" * 60)
    print("Clerk Local-RAG Interactive Query CLI")
    print("=" * 60)

    engine = RAGEngine()
    if engine.embeddings is None:
        print("[ERROR] Database contains no embeddings. Please run embedding_creation.py first.")
        return

    test_queries = [
        "What is the direction or concept of this report?",
        "What are the recommendations regarding human review?",
    ]

    for q in test_queries:
        print(f"\n--- Testing Query: '{q}' ---")
        result = engine.query(q, top_k=3)
        print(f"\n[Answer]:\n{result['answer']}\n")
        print("[Retrieved Context]:")
        for i, ctx in enumerate(result["context_items"], start=1):
            print(f"  {i}. (Page {ctx['page_number']}, Score: {ctx['score']:.4f}): {ctx['sentence_chunk'][:120]}...")

    print("\nEntering interactive mode. Type your question or 'exit' to quit:")
    while True:
        try:
            user_input = input("\nQuery > ").strip()
            if not user_input or user_input.lower() in ("exit", "quit"):
                break
            result = engine.query(user_input, top_k=3)
            print(f"\n[Answer]:\n{result['answer']}")
            print("\n[Sources]:")
            for ctx in result["context_items"]:
                print(f" - Page {ctx['page_number']} (Score: {ctx['score']:.4f})")
        except (KeyboardInterrupt, EOFError):
            break

    print("\nExiting Clerk RAG CLI.")


if __name__ == "__main__":
    main()
