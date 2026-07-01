"""Telegram bot for Cat or Bread classifier"""
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
import asyncio

from model.predict import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    logger.warning("BOT_TOKEN not set. Bot will not start. Set with: export BOT_TOKEN='your_token'")

dp = Dispatcher()
router = Router()

bot_instance = None

def get_bot():
    global bot_instance
    if bot_instance is None:
        if not BOT_TOKEN:
            raise ValueError("BOT_TOKEN not set")
        bot_instance = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
    return bot_instance

def get_start_keyboard():
    kb = [
        [InlineKeyboardButton(
            text="Open Mini App",
            web_app={"url": os.getenv("WEBAPP_URL", "https://catorbread.onrender.com")}
        )],
        [InlineKeyboardButton(text="How it works", callback_data="how_it_works")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=kb)

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "<b>Cat or Bread?</b>\n\n"
        "Send me a photo and I'll tell you if it's a cat or a loaf of bread!\n\n"
        "Just send any image, or open the Mini App below.",
        reply_markup=get_start_keyboard()
    )

@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "How to use:\n\n"
        "1. Send me a photo of a cat or bread\n"
        "2. I'll analyze it with AI\n"
        "3. You'll get a result with confidence percentage\n\n"
        "The model was trained on thousands of cat and bread images!"
    )

@router.callback_query(F.data == "how_it_works")
async def callback_how(callback: CallbackQuery):
    await callback.message.edit_text(
        "<b>How it works</b>\n\n"
        "I use a neural network (ResNet18) trained to distinguish cats from bread.\n\n"
        "The model looks at shapes, textures, colors, and patterns to decide:\n"
        "- Fluffy + whiskers + pointy ears = Cat\n"
        "- Brown + crust + loaf shape = Bread\n\n"
        "Send me a photo to try it!",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="Back", callback_data="back")]]
        )
    )

@router.callback_query(F.data == "back")
async def callback_back(callback: CallbackQuery):
    await callback.message.edit_text(
        "<b>Cat or Bread?</b>\n\n"
        "Send me a photo and I'll tell you if it's a cat or a loaf of bread!\n\n"
        "Just send any image, or open the Mini App below.",
        reply_markup=get_start_keyboard()
    )

@router.message(F.photo)
async def handle_photo(message: Message):
    bot = get_bot()
    processing_msg = await message.reply("Analyzing your image...")

    try:
        file = await bot.get_file(message.photo[-1].file_id)
        image_bytes = await bot.download_file(file.file_path)
        image_data = image_bytes.read()

        result = predict(image_data)

        if "error" in result:
            await processing_msg.edit_text(f"Error: {result['error']}")
            return

        label = result["label"]
        desc = result["description"]
        confidence = result["confidence"]

        bar_cat = "▓" * int(result["probabilities"]["cat"] / 10) + "░" * (10 - int(result["probabilities"]["cat"] / 10))
        bar_bread = "▓" * int(result["probabilities"]["bread"] / 10) + "░" * (10 - int(result["probabilities"]["bread"] / 10))

        response = (
            f"<b>It's a {label}!</b> ({desc})\n\n"
            f"Confidence: <b>{confidence}%</b>\n\n"
            f"Cat   {bar_cat}  {result['probabilities']['cat']}%\n"
            f"Bread {bar_bread}  {result['probabilities']['bread']}%"
        )

        await processing_msg.edit_text(response)

    except Exception as e:
        logger.error(f"Error processing photo: {e}")
        await processing_msg.edit_text("Sorry, I couldn't process that image. Try another one.")

@router.message()
async def handle_other(message: Message):
    if message.text and not message.text.startswith("/"):
        await message.reply(
            "Send me a photo, not text! Or use /start to see the options."
        )

async def main():
    dp.include_router(router)
    bot = get_bot()
    logger.info("Starting Cat or Bread bot...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
