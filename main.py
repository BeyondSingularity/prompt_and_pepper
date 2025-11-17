import asyncio
from os import getenv

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.types import Message
from dotenv import load_dotenv

# Import LLM functions
from llm import RAGService, setup_database


async def health(message: Message):
    await message.answer("All systems operational!")


async def LLM_answer_stream(message: Message):
    """Handle user questions with streaming response (recommended)."""
    user_query = message.text

    # Send typing indicator
    await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    # Send initial message that we'll edit
    response_message = await message.answer("Thinking...")

    # Stream the answer
    full_response = ""
    chunk_buffer = ""

    for chunk in RAGService().query_stream(user_query):
        full_response += chunk
        chunk_buffer += chunk

        # Update message every ~50 characters for smooth streaming
        if len(chunk_buffer) >= 50:
            try:
                await response_message.edit_text(full_response)
                chunk_buffer = ""
            except Exception:
                # Ignore errors if message didn't change enough
                pass

    # Final update with complete response
    try:
        if chunk_buffer:
            await response_message.edit_text(full_response)
    except Exception:
        pass


async def main():
    load_dotenv()
    setup_database(force_rebuild=False)
    bot = Bot(token=getenv("BOT_TOKEN"))
    dp = Dispatcher()

    # Register handlers
    dp.message.register(health, F.text == "/health")

    # Choose ONE of these two handlers:
    # Option 1: Non-streaming (simple, but user waits for complete answer)
    # dp.message.register(LLM_answer, F.text)

    # Option 2: Streaming (better UX, shows answer being generated)
    dp.message.register(LLM_answer_stream, F.text)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
