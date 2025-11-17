"""
One-time database setup script.
Run this to initialize the vector database with recipe embeddings.
"""

import os

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "recipes")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))

DATASET_NAME = os.getenv("DATASET_NAME", "AkashPS11/recipes_data_food.com")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")
MAX_RECIPES = int(os.getenv("MAX_RECIPES", "0"))


def setup_database(force_rebuild: bool = False):
    """
    Initialize ChromaDB with recipe embeddings.

    Args:
        force_rebuild: If True, delete existing collection and rebuild from scratch
    """
    print(f"Initializing database at {CHROMA_PATH}...")

    # Connect to persistent ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Check if collection exists
    try:
        collection = client.get_collection(COLLECTION_NAME)
        if not force_rebuild:
            count = collection.count()
            print(f"✓ Collection '{COLLECTION_NAME}' already exists with {count} recipes")
            print("Use force_rebuild=True to rebuild from scratch")
            return
        else:
            print(f"Deleting existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Create new collection
    print(f"Creating new collection '{COLLECTION_NAME}'...")
    collection = client.create_collection(COLLECTION_NAME)

    # Load dataset
    print(f"Loading dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    # Determine how many recipes to process
    max_recipes = MAX_RECIPES if MAX_RECIPES > 0 else len(dataset)
    max_recipes = min(max_recipes, len(dataset))
    print(f"Processing {max_recipes} recipes...")

    # Load embedding model
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Process in batches to avoid memory issues
    batch_size = 100
    recipe_id = 0

    for i in tqdm(range(0, max_recipes, batch_size), desc="Creating embeddings"):
        end_idx = min(i + batch_size, max_recipes)
        batch_data = dataset[i:end_idx]

        # Build full recipe text combining name, ingredients, and instructions
        texts = []
        metadatas = []

        for idx in range(len(batch_data["RecipeId"])):
            # Extract fields
            name = batch_data["Name"][idx] or "Untitled Recipe"
            ingredients = batch_data["RecipeIngredientParts"][idx] or []
            instructions = batch_data["RecipeInstructions"][idx] or ""

            # Build combined text document
            ingredients_text = ", ".join(ingredients) if ingredients else "No ingredients listed"
            instructions_text = instructions if instructions else "No instructions provided"

            recipe_text = f"Recipe: {name}\n\nIngredients: {ingredients_text}\n\nInstructions: {instructions_text}"

            texts.append(recipe_text)
            metadatas.append({
                "recipe_id": str(batch_data["RecipeId"][idx]),
                "name": name
            })

        # Generate embeddings
        embeddings = embedder.encode(texts, show_progress_bar=False)

        # Add to collection
        collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=[f"recipe-{recipe_id + idx}" for idx in range(len(texts))]
        )

        recipe_id += len(texts)

    final_count = collection.count()
    print(f"\n✓ Database setup complete! Added {final_count} recipes")


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    setup_database(force_rebuild=force)
