# Prompt & Pepper 🌶️

RAG-based recipe assistant powered by LLM.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   Add `BOT_TOKEN` to your `.env` file for Telegram bot.

3. **Initialize database (one-time):**
   ```bash
   python -m llm.setup_db
   ```
   
   This downloads the recipe dataset and creates embeddings. Takes ~5-10 minutes depending on your `MAX_RECIPES` setting.
   
   To rebuild the database:
   ```bash
   python -m llm.setup_db --force
   ```

## Usage

### Standalone (CLI)
```python
from llm import llm_answer, llm_answer_stream

# Simple query
answer = llm_answer("How to cook chocolate cookies?")
print(answer)

# Streaming query
for chunk in llm_answer_stream("What's a good pasta recipe?"):
    print(chunk, end="", flush=True)
```

See `example_usage.py` for more examples.

### Telegram Bot
```bash
python bot.py
```

The bot has two response modes (configured in `bot.py`):
- **Non-streaming**: Waits for complete answer, then sends it
- **Streaming**: Shows answer being generated in real-time (better UX)

## Configuration

Edit `.env` to customize:
- `EMBEDDING_MODEL`: Sentence transformer model for embeddings
- `LLM_MODEL`: Ollama model name (must be installed locally)
- `CHROMA_PATH`: Path to vector database
- `TOP_K_RESULTS`: Number of recipes to use as context
- `MAX_RECIPES`: Limit dataset size (0 = all recipes)

## Project Structure

```
├── llm/                    # LLM module
│   ├── __init__.py        # Public API exports
│   ├── config.py          # Configuration management
│   ├── setup_db.py        # Database initialization
│   └── rag_service.py     # RAG service with streaming
├── bot.py                 # Telegram bot
├── example_usage.py       # CLI usage examples
├── main.py                # Original prototype (deprecated)
├── .env                   # Configuration
├── requirements.txt       # Dependencies
└── chroma_db/            # Vector database (created after setup)
```

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed with your chosen model (e.g., `ollama pull gemma2`)