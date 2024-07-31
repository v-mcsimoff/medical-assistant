import asyncio

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from dotenv import load_dotenv

from main import (
    complete_pipeline
)


load_dotenv()

bot = Bot(token=os.getenv('TELEGRAM_TOKEN'))
dp = Dispatcher()

# Dictionary for the chat history.
conversation_history = {}

# Function to trim the chat history.
def trim_history(history, max_length=15000):
    current_length = sum(len(message["content"]) for message in history)
    while history and current_length > max_length:
        removed_message = history.pop(0)
        current_length -= len(removed_message["content"])
    return history