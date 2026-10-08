import os
import httpx
import re

API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

API_KEY = os.getenv("YANDEX_API_KEY")
FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

ALLOWED_TAGS = {"b", "s", "i", "u", "code"}

HEADERS = {
    "Authorization": f"Api-Key {API_KEY}",
    "Content-Type": "application/json",
}


def convert_markdown_to_html(text: str) -> str:
    """
    Конвертирует markdown-разметку (**bold**, *italic*/_italic_) в HTML-теги,
    на случай если модель случайно вернула markdown вместо HTML.
    """
    # **bold** -> <b>bold</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # *italic* или _italic_ -> <i>italic</i> (только если не часть **, уже обработанного выше)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"_(.+?)_", r"<i>\1</i>", text)
    return text


def sanitize_html_for_telegram(text: str) -> str:
    """
    Убирает все HTML-теги, кроме разрешённых Telegram'ом.
    Если модель случайно вернула неподдерживаемый тег — он будет вырезан,
    но текст внутри останется.
    """
    def replace_tag(match):
        tag_name = match.group(2).lower()
        if tag_name in ALLOWED_TAGS:
            return match.group(0)  # оставляем тег как есть
        return ""  # вырезаем неподдерживаемый тег, текст внутри остаётся

    # Ищем открывающие и закрывающие теги вида <tag> или </tag>
    pattern = r"(<(/?)([a-zA-Z0-9]+)[^>]*>)"

    def replace_match(match):
        full_tag, closing, tag_name = match.group(1), match.group(2), match.group(3)
        if tag_name.lower() in ALLOWED_TAGS:
            return full_tag
        return ""

    return re.sub(pattern, replace_match, text)


async def translate_to_english(text: str) -> str:
    """
    Переводит русский текст на английский (для отображения в скобках).
    """
    prompt = f"""Translate the following Russian text to natural, conversational English. 
Output ONLY the translation, nothing else — no explanations, no quotes.

Text: "{text}"
"""
    messages = [{"role": "user", "text": prompt}]
    raw_result = await _call_yandexgpt(messages)
    return raw_result.strip()

async def _call_yandexgpt(messages: list[dict], temperature: float = 0.6) -> str:
    """
    messages — список вида [{"role": "system"/"user"/"assistant", "text": "..."}]
    """
    payload = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": 500
        },
        "messages": messages
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()

    return data["result"]["alternatives"][0]["message"]["text"]


async def analyze_errors(transcript: str) -> str:
    prompt = f"""You are an English tutor. The user is a non-native speaker practicing English via voice messages.

Show the transcript with corrections using ONLY these HTML tags: <s> for strikethrough, <b> for bold.
Do NOT use any other HTML tags.
Do NOT add any comments, explanations, or meta-text like "no mistakes found" or "looks good".

If there are mistakes: show the text with <s>wrong</s> <b>correct</b> inline.
If there are NO mistakes: just output the original transcript exactly as is, with no tags and no extra text.

Output nothing except the resulting text itself.

Transcript: "{transcript}"
"""
    messages = [{"role": "user", "text": prompt}]
    raw_result = await _call_yandexgpt(messages)
    raw_result = convert_markdown_to_html(raw_result)
    return sanitize_html_for_telegram(raw_result)


async def generate_reply(transcript: str, history: list[dict]) -> str:
    system_prompt = (
        "You are a friendly English conversation partner helping someone "
        "practice spoken English. Keep responses natural and short (2-4 sentences). "
        "Ask follow-up questions. Adapt vocabulary to the user's level. "
        "Do not correct mistakes here — that's handled separately."
    )

    messages = [{"role": "system", "text": system_prompt}]

    for msg in history:
        messages.append({"role": msg["role"], "text": msg["text"]})

    messages.append({"role": "user", "text": transcript})

    return await _call_yandexgpt(messages)


async def detailed_analysis(transcript: str, corrections: str) -> str:
    """
    Даёт развёрнутое объяснение ошибок на русском языке: почему это ошибка,
    какое правило, примеры правильного использования.
    """
    prompt = f"""You are an English tutor helping a Russian-speaking learner.

Below is a transcript from the learner and a short correction you already gave.

Now provide a DETAILED explanation for each mistake, written in RUSSIAN:
- What was wrong and why (grammar rule, common confusion, etc.) — explained in Russian
- A simple explanation in Russian that the learner can understand
- One or two extra EXAMPLE SENTENCES in ENGLISH showing correct usage (examples stay in English, explanations in Russian)

Use ONLY these HTML tags: <b> for bold, <i> for italic. No other tags.
Keep it well-structured but not overly long — a few sentences per mistake.
If there were no real mistakes, briefly note in Russian what was good about the phrasing instead.

Original transcript: "{transcript}"
Short correction given earlier: "{corrections}"
"""
    messages = [{"role": "user", "text": prompt}]
    raw_result = await _call_yandexgpt(messages)
    raw_result = convert_markdown_to_html(raw_result)
    return sanitize_html_for_telegram(raw_result)