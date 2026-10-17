<div align="center" style="display: flex; align-items: center; justify-content: center; gap: 24px;">
  <img src="./data/logo/clerk-logo.png" alt="clerk logo" width="120">
  <div>
    <h1>clerk</h1>
    <p>clerk is that front-desk lady at any organization, but my clerk is a local RAG that I built from scratch.</p>
  </div>
</div>

## Demo 
![Demo Video](./data/demo/demo.gif)


## Motivation

- I was curious about how Retrieval-Augmented Generation (RAG) systems work and wanted to implement one from scratch, given my background in Computer Science with a specialization in AI.
- I was inspired by this [YouTube video](https://youtu.be/qN_2fnOPY-M?si=jYZVkgYIlTdL3E1q), which sparked the idea for this project.
- Most RAG projects use a traditional web application as the interface for submitting queries. I wanted to explore a different approach by building a RAG system that users can interact with through Discord.
- Ever since I started out with python I've always used pip for the package manager and never used UV, and I was told uv is much more faster than pip, so I wanted to try it out myself.

## Prerequisites

- **Python**: `3.10.0` (required)
- **uv**: Modern, ultra-fast Python package and project manager ([Install uv](https://docs.astral.sh/uv/getting-started/installation/))
- **Ollama**: Installed locally with your chosen model (e.g., `gemma4:e2b`)
- **Discord Bot**: A registered Discord application bot with **Message Content Intent** enabled

---

## Installation & Setup

### 1. Clone the Repository

```bash
# Clone the git repository from github
git clone https://github.com/melwin-11/clerk.git
cd clerk
```

### 2. Set Up Virtual Environment with `uv`

This project is fully managed using `uv`. Make sure your environment is created with **Python 3.10.0**:

```bash
# Create a virtual environment using Python 3.10.0
uv venv --python 3.10.0

# Synchronize and install all locked dependencies (including PyTorch CUDA)
uv sync
```

### 3. Configure Environment Variables (`.env`)

Before running any services or the Discord bot, create your local `.env` file from `.env.example`:

```bash
# On Linux / macOS
cp .env.example .env

# On Windows (PowerShell)
Copy-Item .env.example .env
```

Open `.env` and fill in your configuration:

```env
# HuggingFace Token (optional)
HF_TOKEN=

# Device Settings
# Keeping query embedding on CPU frees 100% of GPU VRAM exclusively for Ollama LLM
RAG_EMBEDDING_DEVICE=cpu

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma4:e2b
OLLAMA_NUM_CTX=2048
OLLAMA_TEMPERATURE=0.3

# Discord Bot Credentials
DISCORD_TOKEN=your_discord_bot_token_here
DISCORD_CHANNEL_ID=your_discord_channel_id_here
```

> Ensure **Message Content Intent** is toggled **ON** in your [Discord Developer Portal](https://discord.com/developers/applications) under `Bot -> Privileged Gateway Intents`.

### 4. Start Ollama

Ensure Ollama is running and has your target model downloaded:

```bash
# Start Ollama service (if not already running as a daemon)
ollama serve

# Pull your model
ollama pull gemma4:e2b
```

---

## Running Clerk

> Only run the Discord bot **after** verifying that `.env` has been created and filled according to `.env.example`.

### Option A: Complete End-to-End Pipeline (Recommended)

`main.py` runs the systematic loop: it verifies the PDF in `data/`, detects any document changes via SHA-256 hash comparison, automatically processes embeddings into SQLite if modified, and boots the Discord bot:

```bash
uv run python main.py
```

### Option B: Run the Discord Bot Directly

If your SQLite database is already built (`data/clerk_rag.db`), you can launch the Discord bot directly:

```bash
uv run python src/bot/bot.py
```

### Option C: Standalone Terminal CLI (No Discord Required)

You can also test queries directly in your terminal without Discord:

```bash
uv run python src/rag/rag.py
```

---

## Interacting with the Discord Bot

Once the bot is online in your specified Discord channel:

| Command | Action |
| :--- | :--- |
| `/query <question>` | Initiates an interactive RAG session and answers your query with context passages and page citations. |
| *(text)* | Once a session is active, any regular message in the channel is treated as a follow-up question. |
| `/stop` | Ends the current interactive session. |
| `/status` or `/help` | Displays currently loaded document chunks, active model, and session state. |
