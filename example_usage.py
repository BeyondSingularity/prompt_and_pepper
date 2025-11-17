"""
Example of using the LLM module standalone (without Telegram bot).
"""

from llm import llm_answer, llm_answer_stream


def example_simple():
    """Simple non-streaming example."""
    print("=== Simple Query ===")
    answer = llm_answer("How to cook chocolate cookies?")
    print(answer)
    print()


def example_streaming():
    """Streaming example - shows response as it's generated."""
    print("=== Streaming Query ===")
    query = "What's a good recipe for pasta?"

    print(f"Q: {query}")
    print("A: ", end="", flush=True)

    for chunk in llm_answer_stream(query):
        print(chunk, end="", flush=True)

    print("\n")


def example_custom_context():
    """Example with custom number of context recipes."""
    print("=== Custom Context (5 recipes) ===")
    answer = llm_answer("How to make bread?", top_k=5)
    print(answer)
    print()


if __name__ == "__main__":
    print("Make sure you've run 'python -m llm.setup_db' first!\n")

    example_simple()
    example_streaming()
    example_custom_context()
