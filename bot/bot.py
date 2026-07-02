"""Telegram bot for Cat or Bread classifier"""
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, BufferedInputFile
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
import asyncio

from model.predict import predict
from model.generate import generate_cat_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    logger.warning("BOT_TOKEN not set. Bot will not start. Set with: export BOT_TOKEN='your_token'")

dp = Dispatcher()
router = Router()

bot_instance = None
admin_ids = set()

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
        [InlineKeyboardButton(text="Зачем это", callback_data="how_it_works")],
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

@router.message(Command("generate"))
async def cmd_generate(message: Message):
    bot = get_bot()
    processing = await message.reply("Генерирую котика...")

    image_bytes = generate_cat_bytes()
    if image_bytes is None:
        await processing.edit_text(
            "Генератор ещё не обучен. Запусти `python3 train_gan.py` чтобы обучить."
        )
        return

    await processing.delete()
    await bot.send_photo(
        message.chat.id,
        photo=BufferedInputFile(image_bytes, filename="cat.jpg"),
        caption="Вот сгенерированный котик!"
    )

@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "Как пользоваться:\n\n"
        "1. Отправь фото кота или хлеба — нейросеть определит, кто это\n"
        "2. Используй /generate — нейросеть нарисует нового котика\n\n"
        "Модели обучены на тысячах изображений!"
    )

@router.message(Command("admin"))
async def cmd_admin(message: Message):
    user_id = message.from_user.id
    if user_id in admin_ids:
        admin_ids.remove(user_id)
        await message.reply("Режим администратора выключен.")
    else:
        admin_ids.add(user_id)
        await message.reply(
            "Режим администратора включён. Все фото от других пользователей "
            "будут приходить сюда.\n"
            "Введи /admin ещё раз чтобы выключить."
        )

@router.callback_query(F.data == "how_it_works")
async def callback_how(callback: CallbackQuery):
    await callback.message.edit_text(
        "<b>Для чего это нужно?</b>\n\n"
        "В наше время мы часто сталкиваемся с проблемой, что котики становятся всё больше похожи на сладкие булочки, и чтобы их отличать был создан этот бот, а то странно будет выглядеть если вы укусите кота и погладите хлеб!",
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

        if not is_other:
            def bar(v):
                return "▓" * int(v / 10) + "░" * (10 - int(v / 10))

            p = result["probabilities"]
            response = (
                f"<b>Это {label}!</b>\n\n"
                f"Уверенность: <b>{confidence}%</b>\n\n"
                f"Кот    {bar(p['cat'])}  {p['cat']}%\n"
                f"Хлеб   {bar(p['bread'])}  {p['bread']}%\n"
                f"Другое {bar(p['other'])}  {p['other']}%"
            )
        else:
            response = "<b>И чё ты мне скинул?</b> Отправь фото кота или хлеба"

        if admin_ids:
            sender = message.from_user
            sender_name = sender.full_name or sender.username or f"id{sender.id}"
            mention = f"<a href='tg://user?id={sender.id}'>{sender_name}</a>"
            for admin_id in list(admin_ids):
                if admin_id == message.chat.id:
                    continue
                try:
                    caption = f"📸 Фото от {mention}\n\n{response}"
                    await bot.send_photo(
                        admin_id,
                        photo=BufferedInputFile(image_data, filename="photo.jpg"),
                        caption=caption,
                    )
                except Exception as e:
                    logger.error(f"Failed to forward to admin {admin_id}: {e}")
            if message.chat.id not in admin_ids:
                await message.reply("📸 Фото отправлено администратору.")

        if is_other:
            await processing_msg.delete()
            await bot.send_sticker(
                message.chat.id,
                sticker="CAACAgIAAxkBAAEFASJqRS1bRHJa4veOggx56dEgrdWRswACp3EAAh5t8EgXiDMpQ5GFwzwE"
            )
            await message.reply(response)
            return
        else:
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
