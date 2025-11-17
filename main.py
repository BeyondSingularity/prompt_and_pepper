from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import ollama


client = chromadb.Client()
collection = client.create_collection("recipes")

dataset = load_dataset("AkashPS11/recipes_data_food.com", split="train")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

for item in dataset[0]:
    print(item)


texts = [item for item in dataset["RecipeInstructions"][:500]]  # возьми первые 500 для теста
embeddings = embedder.encode(texts)

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[f"recipe-{i}" for i in range(len(texts))]
)

def rag_query(query: str, top_k: int = 3, model: str = "gemma2"):
    query_emb = embedder.encode([query])[0]
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    context = "\n\n".join(results["documents"][0])
    # print(context)
    prompt = f"""
Ты — помощник, который отвечает на вопросы о рецептах.
Используй приведённые рецепты как контекст:

{context}

Вопрос: {query}
Ответ:
"""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


answer = rag_query("How to cook chocolate cookies?", model="gemma2")
print(answer)