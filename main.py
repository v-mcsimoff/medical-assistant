import os
import base64
import json
import asyncio
import aiogram
import torch

from typing import Optional
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateBeamDecoderOnlyOutput
)
from openai import OpenAI

from dotenv import load_dotenv

from huggingface_hub import login

load_dotenv()

router = Router()

login(os.getenv('HUGGINGFACE_TOKEN'))

# Bot token
BOT_TOKEN = os.getenv('TELEGRAM_TOKEN')

# OpenAI setup
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Model setup
MEDICAL_ASSISTANT_MODEL = 'McSimoff/llama-3-8b-medical-assistant-v3'
DEVICE = "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MEDICAL_ASSISTANT_MODEL
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MEDICAL_ASSISTANT_MODEL)

# Bot states
class BotStates(StatesGroup):
    WAITING_FOR_LEAFLET = State()
    READY_FOR_QUESTIONS = State()

class LeafletBot:
    def __init__(self, model, tokenizer, client):
        self.model = model
        self.tokenizer = tokenizer
        self.client = client
        self.chat_history = []
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device=DEVICE,
        )

    async def process_leaflet(self, image_path: str) -> str:
        with open(image_path, 'rb') as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """
                        Extract the text from this image of a medicine leaflet.
                        """},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=10000,
        )

        extracted_text = response.choices[0].message.content
        self.chat_history = [{
            "role": "system",
            "content": f"Leaflet content: {extracted_text}"
        }]
        messages = [{
            "role": "user",
            "content": f"""
                You are an experienced medical doctor. I need to understand 
                how to use medicine. I don't need information about the 
                leaflet, manufacturer or their contacts, include only 
                information about the medicine in your response. Provide me 
                a summary of this medicine leaflet in under 250 words, 
                including name, uses, from 3 to 6 most important side effects, 
                and recommendations for taking the medicine - in the morning 
                or in the evening, before or after eating, interaction with 
                other medicines or alcohol:
                {extracted_text}
            """
        }]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = self.pipe(
            prompt,
            max_new_tokens=350,
            do_sample=True,
            temperature=0.5,
            top_k=50,
            top_p=0.95
        )

        summary = outputs[0]["generated_text"]
        assistant_response = summary.split(
            '<|im_start|>assistant\n'
        )[1].split('<|im_end|>')[0].strip()

        return assistant_response

    async def process_question(self, question: str) -> str:
        messages = self.chat_history + [{"role": "user", "content": """
            You are an experienced medical doctor. I have sent you a medicine 
            leaflet. I need to undertand how to take the medicine. I have a 
            question about the medicine from the leaflet. Answer the question 
            using information from the leaflet. If you don't know the answer, 
            say that I should ask a doctor. Don't include information about 
            the leaflet or manufacturer, only the answer to the question. Be 
            short, but helpful. Question:
            """ + question}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = self.pipe(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.5,
            top_k=50,
            top_p=0.95
        )
        answer = outputs[0]["generated_text"]
        assitant_answer = answer.split(
            '<|im_start|>assistant\n'
        )[1].split('<|im_end|>')[0].strip()
        return assitant_answer

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()

# Initialize LeafletBot
leaflet_bot = LeafletBot(model, tokenizer, client)

@router.message(Command(commands=['start', 'help']))
async def send_welcome(message: Message, state: FSMContext):
    await message.reply(
        "Welcome! Please send me a photo of a medicine leaflet to get started."
    )
    await state.set_state(BotStates.WAITING_FOR_LEAFLET)

@router.message(
    BotStates.WAITING_FOR_LEAFLET,
    F.content_type.in_({'image', 'photo'})
)
async def handle_leaflet(message: Message, state: FSMContext):
    # Download the photo
    photo = message.photo[-1]
    file_id = photo.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path

    local_filename = f"leaflet_{message.from_user.id}.jpg"
    await bot.download_file(file_path, local_filename)

    # Process the leaflet
    await message.reply("Processing the leaflet. This may take a moment...")
    try:
        summary = await leaflet_bot.process_leaflet(local_filename)
        await message.reply(
            """
            Here's a summary of the most important 
            information from the leaflet:
            """
        )
        await message.reply(summary)
        await message.reply("You can now ask questions about this medicine.")
        await state.set_state(BotStates.READY_FOR_QUESTIONS)
    except Exception as e:
        await message.reply(
            f"An error occurred while processing the leaflet: {str(e)}"
        )
        await state.set_state(BotStates.WAITING_FOR_LEAFLET)

@router.message(BotStates.READY_FOR_QUESTIONS)
async def handle_question(message: Message, state: FSMContext):
    question = message.text
    await message.reply("Processing your question. This may take a moment...")
    response = await leaflet_bot.process_question(question)
    await message.reply(response)

# Add the router to the dispatcher
dp.include_router(router)

async def main():
    # Start the bot
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
