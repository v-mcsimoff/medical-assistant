import os
import base64
import json
import asyncio
import aiogram
from typing import Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
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
MEDICAL_ASSISTANT_MODEL = 'McSimoff/llama-3-8b-medical-assistant-v2'
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
        self.current_leaflet = None
        self.chat_history = []

    async def extract_text(self, image_path: str) -> dict:
        with open('leaflet_schema.json', 'r') as file:
            leaflet_schema = json.load(file)

        with open(image_path, 'rb') as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """
                            Provide JSON file that represents this document. 
                            In the 'leafletText' part provide the whole text 
                            of the leaflet. Use this JSON Schema: 
                            """ + json.dumps(leaflet_schema)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,
        )

        return json.loads(response.choices[0].message.content)

    async def create_prompt(
        self, content: str, is_leaflet: bool
    ) -> tuple[str, dict]:
        if is_leaflet:
            return self._create_prompt_leaflet(content)
        else:
            return self._create_prompt_question(content)

    def _create_prompt_leaflet(self, leaflet: dict) -> tuple[str, dict]:
        system_message = """
            You are a helpful experienced medical doctor, who helps patients 
            to take medicine in a right way. You receive a text of a medicine 
            leaflet and provide a patient with the most important information 
            about the medicine, including its name, uses, the most important 
            and wide-spread side effects, prescriptions, and short and simple 
            instructions to use the medicine.
        """
        
        user_message = """
        Provide me with brief instructions for use of this medicine: 
        """

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": user_message + leaflet['properties']['leafletText']
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        encoded_input = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        return prompt, encoded_input

    def _create_prompt_question(self, question: str) -> tuple[str, dict]:
        system_message = """
        You are a helpful experienced medical doctor, who helps patients to 
        take medicine in a right way. You answer patient's questions about 
        the medicine from the leaflet that the patient has sent you before. 
        If there have been several leaflets sent you should use the last one.
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        history = " ".join([f"{role}: {content}" for role, content in self.chat_history])
        prompt = f"{history} {prompt}"

        encoded_input = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        return prompt, encoded_input

    async def model_response(self, prompt: str, encoded_input: dict) -> list[str]:
        gen_kwargs = {"max_new_tokens": 2, "penalty_alpha": 0.2, "top_k": 2}
        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        model_output = self.model.generate(
            encoded_input['input_ids'].to(DEVICE),
            attention_mask=encoded_input['attention_mask'].to(DEVICE),
            **gen_kwargs,
            streamer=streamer,
        )

        if isinstance(model_output, (
            GenerateDecoderOnlyOutput,
            torch.Tensor,
            GenerateBeamDecoderOnlyOutput
        )):
            response_tokens = model_output.sequences if hasattr(
                model_output,
                'sequences'
            ) else model_output
        else:
            raise ValueError(
                f"Unexpected model output type: {type(model_output)}"
            )

        responses_txt = self.tokenizer.batch_decode(
            response_tokens[:,len(encoded_input['input_ids'][0]):],
            skip_special_tokens=True
        )

        return responses_txt

    async def process_leaflet(self, image_path: str) -> str:
        try:
            self.current_leaflet = await self.extract_text(image_path)
            prompt, encoded_input = await self.create_prompt(
                self.current_leaflet,
                is_leaflet=True
            )
            response = await self.model_response(prompt, encoded_input)
            self.chat_history.append(("bot", response[0]))
            return response[0]
        except Exception as e:
            return f"An error occurred while processing the leaflet: {str(e)}"

    async def process_question(self, question: str) -> str:
        if not self.current_leaflet:
            return "Please send a leaflet image first."
        try:
            prompt, encoded_input = await self.create_prompt(
                question,
                is_leaflet=False
            )
            response = await self.model_response(prompt, encoded_input)
            self.chat_history.append(("user", question))
            self.chat_history.append(("bot", response[0]))
            return response[0]
        except Exception as e:
            return f"An error occurred while processing the question: {str(e)}"

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN)
# storage = MemoryStorage()
dp = Dispatcher()

dp.include_router(router)

# Initialize LeafletBot
leaflet_bot = LeafletBot(model, tokenizer, client)

@router.message(CommandStart())
async def send_welcome(message: Message, state: FSMContext):
    await message.reply("Welcome! Please send me a photo of a medicine leaflet to get started.")
    await state.set_state(BotStates.WAITING_FOR_LEAFLET)

@router.message(BotStates.WAITING_FOR_LEAFLET, F.content_type.in_({'photo', 'image'}))
async def handle_leaflet(message: Message, state: FSMContext):
    # Download the photo
    photo = message.photo[-1]
    file_id = photo.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, 'leaflet.jpg')

    # Process the leaflet
    await message.reply("Processing the leaflet. This may take a moment...")
    response = await leaflet_bot.process_leaflet('leaflet.jpg')
    
    await message.reply(response)
    await message.reply("You can now ask questions about this medicine.")
    await state.set_state(BotStates.READY_FOR_QUESTIONS)

@router.message(BotStates.READY_FOR_QUESTIONS)
async def handle_question(message: Message, state: FSMContext):
    question = message.text
    await message.reply("Processing your question. This may take a moment...")
    response = await leaflet_bot.process_question(question)
    await message.reply(response)

async def main():
    # Start the bot
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
