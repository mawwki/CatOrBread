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
            text="Открыть Mini App",
            web_app={"url": os.getenv("WEBAPP_URL", "https://catorbread.onrender.com")}
        )],
        [InlineKeyboardButton(text="Как это работает", callback_data="how_it_works")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=kb)

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "<b>Кот или Хлеб?</b>\n\n"
        "Отправь фото, и я скажу: это котик или булочка!\n\n"
        "Просто отправь изображение или открой Mini App.",
        reply_markup=get_start_keyboard()
    )

@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "Как пользоваться:\n\n"
        "1. Отправь фото кота или хлеба\n"
        "2. Нейросеть проанализирует его\n"
        "3. Получишь результат с процентом уверенности\n\n"
        "Модель обучена на тысячах изображений котов и выпечки!"
    )

@router.callback_query(F.data == "how_it_works")
async def callback_how(callback: CallbackQuery):
    await callback.message.edit_text(
        "<b>Как это работает</b>\n\n"
        "Я использую нейросеть (ResNet18), обученную отличать котов от хлеба.\n\n"
        "Модель анализирует формы, текстуры, цвета и паттерны:\n"
        "- Пушистость + усы + ушки = Кот\n"
        "- Корочка + форма батона = Хлеб\n\n"
        "Отправь фото и попробуй!",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="Назад", callback_data="back")]]
        )
    )

@router.callback_query(F.data == "back")
async def callback_back(callback: CallbackQuery):
    await callback.message.edit_text(
        "<b>Кот или Хлеб?</b>\n\n"
        "Отправь фото, и я скажу: это котик или булочка!\n\n"
        "Просто отправь изображение или открой Mini App.",
        reply_markup=get_start_keyboard()
    )

@router.message(F.photo)
async def handle_photo(message: Message):
    bot = get_bot()
    processing_msg = await message.reply("Анализирую изображение...")

    try:
        file = await bot.get_file(message.photo[-1].file_id)
        image_bytes = await bot.download_file(file.file_path)
        image_data = image_bytes.read()

        result = predict(image_data)

        if "error" in result:
            await processing_msg.edit_text(f"Ошибка: {result['error']}")
            return

        label = result["label"]
        desc = result["description"]
        confidence = result["confidence"]

        is_other = result["prediction"] == "other"

        if is_other:
            await processing_msg.delete()
            await bot.send_sticker(
                message.chat.id,
                sticker="CAACAgIAAxkBAAEFASJqRS1bRHJa4veOggx56dEgrdWRswACp3EAAh5t8EgXiDMpQ5GFwzwE"
            )
            await message.reply(
                "<b>И чё ты мне скинул?</b> Отправь фото кота или хлеба"
            )
            return
        else:
            bar_cat = "▓" * int(result["probabilities"]["cat"] / 10) + "░" * (10 - int(result["probabilities"]["cat"] / 10))
            bar_bread = "▓" * int(result["probabilities"]["bread"] / 10) + "░" * (10 - int(result["probabilities"]["bread"] / 10))

            response = (
                f"<b>Это {label}!</b> ({desc})\n\n"
                f"Уверенность: <b>{confidence}%</b>\n\n"
                f"Кот   {bar_cat}  {result['probabilities']['cat']}%\n"
                f"Хлеб  {bar_bread}  {result['probabilities']['bread']}%"
            )

        await processing_msg.edit_text(response)

    except Exception as e:
        logger.error(f"Error processing photo: {e}")
        await processing_msg.edit_text("Извини, не смог обработать это изображение. Попробуй другое.")

@router.message()
async def handle_other(message: Message):
    if message.text and not message.text.startswith("/"):
        await message.reply(
            "Отправь фото, а не текст! Используй /start чтобы увидеть команды."
        )

async def main():
    dp.include_router(router)
    bot = get_bot()
    logger.info("Starting Cat or Bread bot...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
