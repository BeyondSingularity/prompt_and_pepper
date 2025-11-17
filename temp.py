from os import getenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def parse_r_list(text):
    """Parse R-style c() list string to Python list."""
    if not text or not isinstance(text, str):
        return []
    # Remove 'c(' prefix and ')' suffix, then split by '", "'
    text = text.strip()
    if text.startswith('c(') and text.endswith(')'):
        text = text[2:-1]  # Remove c( and )
    # Split by comma and clean quotes
    items = []
    for item in text.split('", "'):
        item = item.strip(' "')
        if item:
            items.append(item)
    return items


DATASET_NAME = getenv("DATASET_NAME", "AkashPS11/recipes_data_food.com")
dataset = load_dataset(DATASET_NAME, split="train")
s, f = 0, 4
batch_data = dataset[s:f]
for idx in range(len(batch_data["RecipeId"])):
    name = batch_data["Name"][idx] or "Untitled Recipe"
    ingredients_raw = batch_data["RecipeIngredientParts"][idx] or ""
    instructions_raw = batch_data["RecipeInstructions"][idx] or ""

    # Parse R-style lists
    ingredients = parse_r_list(ingredients_raw)
    instructions = parse_r_list(instructions_raw)

    # Build combined text document
    ingredients_text = "\n- ".join(ingredients) if ingredients else "No ingredients listed"
    instructions_text = "\n".join(instructions) if instructions else "No instructions provided"

    recipe_text = f"Recipe: {name}\n\nIngredients:\n- {ingredients_text}\n\nInstructions:\n{instructions_text}"
    print(recipe_text)
