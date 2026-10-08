import os
import uuid
from pydub import AudioSegment
from aiogram import Bot
from aiogram.types import Voice

# Папка для временных файлов
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)


async def download_voice(bot: Bot, voice: Voice) -> str:
    """
    Скачивает голосовое сообщение из Telegram и сохраняет как .ogg
    Возвращает путь к скачанному файлу.
    """
    file = await bot.get_file(voice.file_id)
    file_path = file.file_path

    local_ogg_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.ogg")
    await bot.download_file(file_path, destination=local_ogg_path)

    return local_ogg_path


def convert_ogg_to_wav(ogg_path: str) -> str:
    """
    Конвертирует .ogg (opus) в .wav — формат, удобный для STT-моделей.
    Возвращает путь к .wav файлу.
    """
    wav_path = ogg_path.replace(".ogg", ".wav")

    audio = AudioSegment.from_file(ogg_path, format="ogg")
    audio.export(wav_path, format="wav")

    return wav_path


def cleanup_files(*paths: str):
    """Удаляет временные файлы после обработки."""
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)