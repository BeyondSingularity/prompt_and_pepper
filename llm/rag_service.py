import os
import sys
from typing import Generator, Optional, AsyncGenerator

import chromadb
import ollama
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer
from llm.utils import Singleton
from llm.conversation_state import Storage

logger.remove()
logger.add(sys.stdout, level="DEBUG")
load_dotenv()


class RAGService(metaclass=Singleton):
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemma2")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "recipes")
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))

    DATASET_NAME = os.getenv("DATASET_NAME", "AkashPS11/recipes_data_food.com")
    DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")
    MAX_RECIPES = int(os.getenv("MAX_RECIPES", "0"))

    def __init__(self):
        self.client = chromadb.PersistentClient(path=RAGService.CHROMA_PATH)

        try:
            self.collection = self.client.get_collection(RAGService.COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Collection '{RAGService.COLLECTION_NAME}' not found. "
                f"Run 'python -m llm.setup_db' first to initialize the database."
            ) from e

        self.embedder = SentenceTransformer(RAGService.EMBEDDING_MODEL)
        self.model = RAGService.LLM_MODEL

    def get_context(self, query: str, top_k: int = None) -> str:
        if top_k is None:
            top_k = RAGService.TOP_K_RESULTS

        query_emb = self.embedder.encode([query])[0]

        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k
        )

        documents = results["documents"][0]
        context = "## " + "\n\n## ".join(documents)

        return context

    async def query_stream(self, query: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        try:
            logger.debug(f"Query sent to LLM:\n{query}")
            async for chunk in ollama.AsyncClient().chat(
                model=self.model,
                messages=query,
                stream=True
            ):
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

        except chromadb.errors.NotEnoughElementsError:
            yield "Error: Not enough recipes in database. Please run setup_db.py first."
        except ollama.ResponseError as e:
            yield f"Error: LLM service unavailable - {e}"
        except Exception as e:
            yield f"Error: {e}"
