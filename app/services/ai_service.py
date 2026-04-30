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

Output: JSON valid saja, tanpa teks lain.

{
  "intent": "expense | income | unknown",
  "amount": number,
  "category": string,
  "description": string
}

Aturan amount (Rupiah, integer):
- "20rb" / "20 ribu" / "20k" -> 20000
- "1.5jt" / "1,5 juta" -> 1500000
- angka polos seperti "2000" -> 2000

EXPENSE — hanya jika pesan mengandung sinyal pengeluaran eksplisit:
beli, bayar, bayarin, habis, keluar, keluarin, belanja, jajan, isi, top up,
kasih ke, transfer ke, kirim ke, langganan, sewa, cicilan, ongkos, gajiin

INCOME — hanya jika pesan mengandung sinyal pemasukan eksplisit:
terima, dapat, dapet, masuk, jual, jualan, laku, dibayar, dibayarin,
dp, omset, pemasukan, transfer dari, kirim dari, dari [nama orang]

UNKNOWN — gunakan ini jika:
- tidak ada sinyal eksplisit di atas
- pesan hanya berisi nama barang + nominal tanpa kata kerja (contoh: "tahu 2000", "bensin 50rb", "kopi 10rb")
- sapaan, pertanyaan, atau bukan transaksi
- JANGAN menebak intent jika ragu — pilih unknown, set amount = 0

category: food, transport, sales, supplies, utilities, salary, rent, dll
description: ringkasan singkat bahasa Indonesia berdasarkan isi pesan"""


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


def transcribe_audio_bytes(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    clean_mime = mime_type.split(";")[0].strip()
    ext = clean_mime.split("/")[-1]
    ext_map = {"webm": "webm", "ogg": "ogg", "oga": "ogg", "mp4": "mp4",
               "mpeg": "mp3", "mpga": "mp3", "wav": "wav", "m4a": "m4a"}
    filename = f"audio.{ext_map.get(ext, 'webm')}"
    transcription = _get_client().audio.transcriptions.create(
        file=(filename, audio_bytes, clean_mime),
        model=AUDIO_MODEL,
        language="id",
    )
    return transcription.text


def process_image_dummy() -> ImageResult:
    return ImageResult(status="valid", message="Gambar valid")
