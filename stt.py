import asyncio
import logging
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu", compute_type="int8")

RU_THRESHOLD = 0.85

# Подсказка с типичными ошибками и паузами, чтобы Whisper не исправлял речь
PROMPT_EN = (
    "Umm, so, I feel very tired today. Yesterday I go to the shop "
    "and buy some things, you know? It was not so good."
)


def _pick_language(wav_path: str) -> str:
    # transcribe() сразу определяет язык и возвращает info,
    # а сами сегменты расшифровываются лениво, поэтому здесь мы их не трогаем
    _, info = model.transcribe(wav_path, language=None)

    probs = dict(info.all_language_probs or [])
    p_en = probs.get("en", 0.0)
    p_ru = probs.get("ru", 0.0)
    logging.info(
        f"Language probs: en={p_en:.2f} ru={p_ru:.2f} (whisper picked: {info.language})"
    )

    if p_ru >= RU_THRESHOLD and p_ru > p_en:
        return "ru"
    return "en"


def _transcribe_sync(wav_path: str) -> tuple[str, str]:
    language = _pick_language(wav_path)

    segments, _ = model.transcribe(
        wav_path,
        language=language,
        initial_prompt=PROMPT_EN if language == "en" else None,
        condition_on_previous_text=False,  # меньше «додумывания» по контексту
        temperature=0.0,                   # без случайности
        beam_size=5,
        vad_filter=True,                   # отрезает тишину, меньше галлюцинаций
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, language



async def transcribe_audio(wav_path: str) -> tuple[str, str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _transcribe_sync, wav_path)