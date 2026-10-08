import asyncio
import logging
import os
import uuid

from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()

from audio_utils import download_voice, convert_ogg_to_wav, cleanup_files
from stt import transcribe_audio
from llm import analyze_errors, generate_reply, detailed_analysis,  translate_to_english


# Хранилище данных для callback-кнопок: {analysis_id: {"transcript": ..., "corrections": ...}}
pending_analysis: dict[str, dict] = {}

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Хранение истории диалога в памяти: {user_id: [{"role": ..., "text": ...}]}
conversation_history: dict[int, list[dict]] = {}
MAX_HISTORY_LENGTH = 10


def add_to_history(user_id: int, role: str, text: str):
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    conversation_history[user_id].append({"role": role, "text": text})

    if len(conversation_history[user_id]) > MAX_HISTORY_LENGTH:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY_LENGTH:]


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Отправь мне голосовое сообщение на английском, "
        "и я помогу тебе с языком 🎙️"
    )


MAX_PENDING_ANALYSIS = 100

def add_pending_analysis(analysis_id: str, data: dict):
    pending_analysis[analysis_id] = data
    if len(pending_analysis) > MAX_PENDING_ANALYSIS:
        # Удаляем самую старую запись (первую по порядку вставки)
        oldest_key = next(iter(pending_analysis))
        del pending_analysis[oldest_key]

@dp.message(F.voice)
async def handle_voice(message: Message):
    user_id = message.from_user.id
    processing_msg = await message.answer("🤔 думаю...")

    if message.voice.duration > 120:
        await message.answer("Слишком длинное сообщение, давай покороче 🙂")
        return

    ogg_path = None
    wav_path = None

    try:
        # Cкачивание и конвертация
        ogg_path = await download_voice(bot, message.voice)
        wav_path = convert_ogg_to_wav(ogg_path)

        # Получаем и текст, и определённый язык
        transcript, detected_language = await transcribe_audio(wav_path)

        if detected_language == "ru":
            # Если сказано по-русски — показываем оригинал + перевод в скобках
            translation = await translate_to_english(transcript)
            await processing_msg.edit_text(f"📍 {transcript}\n({translation}")


        # Дальше анализ ошибок для английской речи 
        if detected_language != "ru":
            corrections = await analyze_errors(transcript)

            analysis_id = str(uuid.uuid4())
            pending_analysis[analysis_id] = {
                "transcript": transcript,
                "corrections": corrections
            }

            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(
                    text="🔍 Подробный разбор ошибок",
                    callback_data=f"detail:{analysis_id}"
                )]
            ])

            await processing_msg.edit_text(
                f"📍 <i>{corrections}</i>",
                parse_mode="HTML",
                reply_markup=keyboard
            )

        # Для генерации ответа используем именно английский вариант текста —
        # если была русская фраза, используем перевод, чтобы диалог оставался на английском
        text_for_reply = translation if detected_language == "ru" else transcript

        history = conversation_history.get(user_id, [])
        reply_text = await generate_reply(text_for_reply, history)

        add_to_history(user_id, "user", text_for_reply)
        add_to_history(user_id, "assistant", reply_text)

        await message.answer(f"💬 {reply_text}")

    except Exception as e:
        logging.exception("Ошибка обработки голосового")
        await message.answer(f"Произошла ошибка: {e}")

    finally:
        cleanup_files(ogg_path, wav_path)
        #await processing_msg.delete()
            


@dp.message()
async def handle_other(message: Message):
    await message.answer("Пожалуйста, отправь голосовое сообщение 🎤")


@dp.callback_query(F.data.startswith("detail:"))
async def handle_detail_button(callback: CallbackQuery):
    analysis_id = callback.data.split(":", 1)[1]

    data = pending_analysis.get(analysis_id)
    if not data:
        await callback.answer("Данные устарели, попробуй отправить новое голосовое 🙁", show_alert=True)
        return

    # Убираем "часики" на кнопке сразу, показываем что процесс пошёл
    await callback.answer("Готовлю подробный разбор...")

    # Можно показать индикатор "печатает..."
    await bot.send_chat_action(callback.message.chat.id, "typing")

    try:
        detail_text = await detailed_analysis(data["transcript"], data["corrections"])
        await callback.message.answer(
            f"🔍 {detail_text}",
            parse_mode="HTML"
        )
    except Exception as e:
        logging.exception("Ошибка при генерации подробного разбора")
        await callback.message.answer(f"Не получилось сделать разбор: {e}")



async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())