import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Document Ingestion Settings
pdf_path = os.getenv("PDF_PATH", "data/monkeybusiness-report.pdf")
sentence_chunk_size = int(os.getenv("SENTENCE_CHUNK_SIZE", "11"))
min_chunk_tokens = int(os.getenv("MIN_CHUNK_TOKENS", "71"))

# Embedding Model Settings
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
embedding_max_seq_length = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "384"))

# Storage Settings
sqlite_db_path = os.getenv("SQLITE_DB_PATH", "data/clerk_rag.db")

# Ollama LLM Settings
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "gemma4:e2b")

# Discord Bot Settings
discord_token = os.getenv("DISCORD_TOKEN", "")
discord_channel_id = os.getenv("DISCORD_CHANNEL_ID", "")