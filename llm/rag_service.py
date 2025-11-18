"""
RAG service for recipe queries with streaming support.
"""

import os
import sys
from typing import Generator, Optional

import chromadb
import ollama
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer
from llm.utils import Singleton
from llm.conversation_state import Storage

# Configure loguru to output to stdout
logger.remove()  # Remove default handler
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
        """Initialize the RAG service with persistent ChromaDB."""
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
        """
        Retrieve relevant recipe context from vector database.

        Args:
            query: User's question
            top_k: Number of recipes to retrieve (default from RAGService)

        Returns:
            Formatted context string with relevant recipes
        """
        if top_k is None:
            top_k = RAGService.TOP_K_RESULTS

        # Generate query embedding
        query_emb = self.embedder.encode([query])[0]

        # Search for similar recipes
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k
        )

        # Format context
        documents = results["documents"][0]
        context = "## " + "\n\n## ".join(documents)

        return context

    def query_stream(self, query: list[dict[str, str]]) -> Generator[str, None, None]:
        """
        Stream the answer to a recipe question token by token.

        Args:
            query: User's question
            top_k: Number of recipes to use as context

        Yields:
            Individual chunks of the response as they are generated
        """
        try:
            logger.debug(f"Query sent to LLM:\n{query}")

            stream = ollama.chat(
                model=self.model,
                messages=query,
                stream=True
            )

            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

        except chromadb.errors.NotEnoughElementsError:
            yield "Error: Not enough recipes in database. Please run setup_db.py first."
        except ollama.ResponseError as e:
            yield f"Error: LLM service unavailable - {e}"
        except Exception as e:
            yield f"Error: {e}"
