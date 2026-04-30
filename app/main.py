import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from app.db import get_transactions, init_db, insert_transaction
from app.services.ai_service import analyze_message, process_image_dummy, transcribe_audio, transcribe_audio_bytes
from app.services.pdf_service import generate_pdf_report

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "c2c_verify")
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="C2C Chat to Core", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def health():
    return {"status": "ok", "service": "C2C Chat to Core"}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/webhook")
def verify_webhook(
    hub_mode: str = Query(alias="hub.mode", default=""),
    hub_verify_token: str = Query(alias="hub.verify_token", default=""),
    hub_challenge: str = Query(alias="hub.challenge", default=""),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def receive_webhook(request: Request):
    body = await request.json()

    try:
        entry = body["entry"][0]
        change = entry["changes"][0]["value"]
        message = change["messages"][0]
    except (KeyError, IndexError):
        return JSONResponse({"status": "ignored"})

    sender = message.get("from", "unknown")
    msg_type = message.get("type", "")

    if msg_type == "text":
        text_body = message["text"]["body"]
        result = analyze_message(text_body)
        if result["intent"] != "unknown":
            insert_transaction(sender, result["intent"], result["amount"],
                               result["category"], result["description"])
        reply = _build_reply(result)

    elif msg_type == "audio":
        audio_id = message["audio"]["id"]
        media_url = f"https://graph.facebook.com/v19.0/{audio_id}"
        transcript = transcribe_audio(media_url)
        result = analyze_message(transcript)
        if result["intent"] != "unknown":
            insert_transaction(sender, result["intent"], result["amount"],
                               result["category"], result["description"])
        reply = f"Transkripsi: \"{transcript}\"\n\n" + _build_reply(result)

    elif msg_type == "image":
        img_result = process_image_dummy()
        reply = f"{img_result['message']} — bukti pembayaran diterima."

    else:
        reply = "Maaf, tipe pesan ini belum didukung. Kirim teks, voice note, atau foto bukti transfer."

    return JSONResponse({"reply": reply})


class SendRequest(BaseModel):
    phone_number: str
    message: str


@app.post("/send")
def send_message(body: SendRequest):
    result = analyze_message(body.message)
    if result["intent"] != "unknown":
        insert_transaction(
            body.phone_number,
            result["intent"],
            result["amount"],
            result["category"],
            result["description"],
        )
    return JSONResponse({"reply": _build_reply(result)})


class AudioRequest(BaseModel):
    phone_number: str
    audio_b64: str
    mime_type: str = "audio/webm"


@app.post("/send-audio")
async def send_audio(body: AudioRequest):
    audio_bytes = base64.b64decode(body.audio_b64)
    transcript = transcribe_audio_bytes(audio_bytes, body.mime_type)
    result = analyze_message(transcript)
    if result["intent"] != "unknown":
        insert_transaction(
            body.phone_number, result["intent"], result["amount"],
            result["category"], result["description"],
        )
    reply = _build_reply(result)
    return JSONResponse({"reply": reply, "transcript": transcript})


@app.get("/transactions/{phone_number}")
def list_transactions(phone_number: str):
    rows = get_transactions(phone_number)
    data = [
        {
            "id": r["id"],
            "intent": r["intent"],
            "amount": r["amount"],
            "category": r["category"],
            "description": r["description"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]
    return JSONResponse({"transactions": data})


@app.get("/report/{phone_number}")
def get_report(phone_number: str):
    path = generate_pdf_report(phone_number)
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"laporan_{phone_number}.pdf",
    )


def _build_reply(result: dict) -> str:
    intent = result.get("intent", "unknown")
    amount = result.get("amount", 0)
    category = result.get("category", "-")
    description = result.get("description", "-")
    amount_str = "Rp " + format(amount, ",d").replace(",", ".")

    if intent == "expense":
        return f"Pengeluaran tercatat: {amount_str} untuk {category}\n{description}"
    elif intent == "income":
        return f"Pemasukan tercatat: {amount_str} dari {category}\n{description}"
    else:
        return "Tidak dapat mengidentifikasi transaksi. Coba lebih spesifik, contoh: \"habis 20rb buat makan\"."
