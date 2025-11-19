import asyncio
from os import getenv

from aiogram import Bot, Dispatcher, F, ParseMode
from aiogram.enums import ChatAction
from aiogram.types import Message
from dotenv import load_dotenv

# Import LLM functions
from llm import RAGService, Storage, setup_database


async def health(message: Message):
    await message.answer("All systems operational!")


async def clear_conversation(message: Message):
    """Clear the conversation history for the user."""
    user_id = str(message.from_user.id)
    Storage().clear_conversation(user_id)
    await message.answer("✅ Ваша история разговоров была очищена.")


async def LLM_answer_stream(message: Message):
    """Handle user questions with streaming response (recommended)."""
    response = await message.answer("⏳ Думаю...")

    user_id = str(message.from_user.id)
    convo = Storage().get_conversation(user_id)
    current_msg = [{"role": "user", "content": message.text}]
    Storage().add_message(user_id, "user", message.text)

    recipes_prompt = "\n---\n".join([m["content"] for m in convo + current_msg])
    print(recipes_prompt)
    recipes = RAGService().get_context(recipes_prompt)

    system_prompt = "Ты — помощник, который отвечает на вопросы о рецептах. " + \
                    "Используй приведённые рецепты как контекст:\n\n" + recipes
    system_prompt = [{"role": "system", "content": system_prompt}]

    full_conversation = system_prompt + convo + current_msg

    full_response = ""
    chunk_buffer = ""
    for chunk in RAGService().query_stream(full_conversation):
        full_response += chunk
        chunk_buffer += chunk

        if len(chunk_buffer) >= 50:
            try:
                await response.edit_text(full_response)
                chunk_buffer = ""
            except Exception:
                pass

    try:
        if chunk_buffer:
            await response.edit_text(full_response, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception:
        pass

    Storage().add_message(user_id, "assistant", full_response)
    print("✓ Response sent to user.")


async def main():
    load_dotenv()
    setup_database(force_rebuild=False)
    bot = Bot(token=getenv("BOT_TOKEN"))
    dp = Dispatcher()

    # Register handlers
    dp.message(F.text == "/health")(health)
    dp.message(F.text == "/clear")(clear_conversation)
    dp.message(F.text)(LLM_answer_stream)

    print("Bot is running...")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
