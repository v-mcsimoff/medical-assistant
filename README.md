# Medical Leaflet Assistant Bot

## Description

This Telegram bot serves as a medical leaflet assistant, capable of processing images of medicine leaflets and providing concise summaries. It uses advanced AI to extract key information from leaflet images and answer follow-up questions about the medication.

## Features

- Process images of medicine leaflets
- Provide summaries including:
  - Medicine name
  - Uses
  - Important side effects
  - Recommendations for taking the medicine
- Answer follow-up questions about the medication
- User-friendly Telegram interface

## Technologies Used

- Python 3.11
- aiogram 3.10 (Telegram Bot API)
- OpenAI GPT-4o-mini API
- The Meta Llama3 model
- Transformers library (Hugging Face)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/medical-leaflet-assistant-bot.git
cd medical-leaflet-assistant-bot
```
2. Install the required packages:
```
pip install .
```
3. Set up your environment variables in a `.env` file:
```
OPENAI_API_KEY = 'your_openai_api_key'
HUGGINGFACE_TOKEN = 'your_huggingface_key'
TELEGRAM_TOKEN = 'your_telegram_bot_key'
```

## Usage

1. Start the bot:
```
python main.py
```
2. In Telegram, start a conversation with the bot.

3. Send an image of a medicine leaflet to the bot.

4. The bot will process the image and provide a summary of the medication.

5. You can then ask follow-up questions about the medication.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Disclaimer

This bot is for informational purposes only and should not be considered as medical advice. Always consult with a healthcare professional before making any decisions about medication.
