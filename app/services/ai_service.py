import json
import os

import httpx
from groq import Groq

from app.schemas import ImageResult, TransactionResult

_client: Groq | None = None
TEXT_MODEL = "llama-3.1-8b-instant"
AUDIO_MODEL = "whisper-large-v3-turbo"


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


SYSTEM_PROMPT = """Kamu adalah parser transaksi keuangan untuk pesan WhatsApp UMKM Indonesia.

Tugas: ubah pesan pengguna menjadi JSON berikut, tanpa teks lain di luar JSON.

{
  "intent": "expense | income | unknown",
  "amount": number,
  "category": string,
  "description": string
}

Aturan:
- amount dalam Rupiah bulat. "20rb" atau "20 ribu" = 20000, "1.5jt" atau "1,5 juta" = 1500000
- category: food, transport, sales, supplies, utilities, salary, dll
- expense: uang keluar (beli, bayar, habis, keluar)
- income: uang masuk (terima, dapat, masuk, jualan, DP)
- tidak jelas atau bukan transaksi: intent = "unknown", amount = 0
- description: ringkasan singkat dalam bahasa Indonesia"""


def analyze_message(user_text: str) -> TransactionResult:
    response = _get_client().chat.completions.create(
        model=TEXT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)
    return TransactionResult(
        intent=str(data.get("intent", "unknown")),
        amount=int(data.get("amount", 0)),
        category=str(data.get("category", "")),
        description=str(data.get("description", "")),
    )


def transcribe_audio(media_url: str) -> str:
    token = os.getenv("WHATSAPP_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"}

    with httpx.Client(timeout=30) as http:
        resp = http.get(media_url, headers=headers)
        resp.raise_for_status()
        audio_bytes = resp.content

    transcription = _get_client().audio.transcriptions.create(
        file=("audio.ogg", audio_bytes, "audio/ogg"),
        model=AUDIO_MODEL,
        language="id",
    )
    return transcription.text


def process_image_dummy() -> ImageResult:
    return ImageResult(status="valid", message="Gambar valid")
