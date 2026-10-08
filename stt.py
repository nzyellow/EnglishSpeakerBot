import asyncio
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")


def _transcribe_sync(wav_path: str) -> tuple[str, str]:
    """
    Возвращает (текст, определённый_язык).
    language=None — Whisper сам определяет язык по аудио.
    """
    segments, info = model.transcribe(wav_path, language=None)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    detected_language = info.language  # 'en', 'ru', и т.д.
    return text, detected_language


async def transcribe_audio(wav_path: str) -> tuple[str, str]:
    loop = asyncio.get_event_loop()
    text, language = await loop.run_in_executor(None, _transcribe_sync, wav_path)
    return text, language