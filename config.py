import os
from dotenv import load_dotenv

load_dotenv()

pdf_path = os.getenv("PDF_PATH", "data/monkeybusiness-report.pdf")
sentence_chunk_size = int(os.getenv("SENTENCE_CHUNK_SIZE", "11"))
min_chunk_tokens = int(os.getenv("MIN_CHUNK_TOKENS", "71"))

embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
embedding_max_seq_length = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "384"))

sqlite_db_path = os.getenv("SQLITE_DB_PATH", "data/clerk_rag.db")

# Device Settings (Run query embedding on CPU to reserve 100% GPU VRAM for Ollama)
rag_embedding_device = os.getenv("RAG_EMBEDDING_DEVICE", "cpu")

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
ollama_num_gpu = int(os.getenv("OLLAMA_NUM_GPU")) if os.getenv("OLLAMA_NUM_GPU") else None
ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))

discord_token = os.getenv("DISCORD_TOKEN", "")
discord_channel_id = os.getenv("DISCORD_CHANNEL_ID", "")