"""EMA Lightning sesli asistan. Zipformer dinler, Gemini cevaplar, EMA okur."""
from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from livekit import api as livekit_api
from pydantic import BaseModel, Field

from agent.config import AGENT_PROMPT, DEFAULT_MODEL, read_env
from agent.judge import JEV_MODEL
from agent.plugins import STT_RATE, decode_pcm, load_ema, load_recognizer

RATES = (48000, 24000, 16000, 8000)

tts = None
device_name = "yükleniyor"
ready = False
stt_ready = False
lock = threading.Lock()


def for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"[*_#>`\[\]]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:600]


def gemini_reply(history: list[dict], user_text: str) -> tuple[str, str]:
    key = read_env("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY .env dosyasında yok.")
    model = read_env("GEMINI_MODEL") or DEFAULT_MODEL
    contents = []
    for turn in history[-8:]:
        role = "user" if turn["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": turn["text"]}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})
    body = {
        "system_instruction": {"parts": [{"text": AGENT_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 512,
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        },
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        if error.code in (400, 401, 403) and "API key" in raw:
            raise HTTPException(status_code=502, detail="Gemini anahtarı geçersiz.") from error
        if error.code == 429:
            raise HTTPException(status_code=502, detail="Gemini kotası doldu.") from error
        raise HTTPException(status_code=502, detail="Gemini isteği reddedildi.") from error
    except urllib.error.URLError as error:
        raise HTTPException(status_code=502, detail="Gemini'ye ulaşılamadı.") from error
    parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    texts = [part["text"] for part in parts if part.get("text") and not part.get("thought")]
    reply = for_speech(" ".join(texts))
    if not reply:
        raise HTTPException(status_code=502, detail="Gemini boş yanıt döndü.")
    return reply, model


def wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype="<f4").astype(np.float32)
    else:
        raise ValueError("WAV 16-bit veya 32-bit olmalı.")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio), rate


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == STT_RATE or audio.size == 0:
        return np.ascontiguousarray(audio, dtype=np.float32)
    source = np.arange(audio.size, dtype=np.float64) / rate
    count = int(round(audio.size * STT_RATE / rate))
    target = np.arange(count, dtype=np.float64) / STT_RATE
    return np.ascontiguousarray(np.interp(target, source, audio), dtype=np.float32)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global tts, device_name, ready, stt_ready
    import torch

    tts = load_ema()
    device_name = torch.cuda.get_device_name(0) if tts.device.type == "cuda" else "CPU"
    ready = True
    load_recognizer()
    stt_ready = True
    yield


app = FastAPI(title="EMA Lightning", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Duration", "X-Seed", "X-Sample-Rate", "X-Generate-Ms"],
)


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=2000)


class AgentIn(BaseModel):
    text: str = Field(max_length=2000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)
    speed: float = 1.0
    seed: int = 0
    sample_rate: int = 48000


class SpeakIn(BaseModel):
    text: str = Field(max_length=8000)
    speed: float = 1.0
    seed: int = 0
    sample_rate: int = 48000


def mask_key(secret: str) -> str:
    return f"…{secret[-4:]}" if len(secret) > 4 else ""


@app.get("/api/health")
def health():
    router_key = read_env("OPENROUTER_API_KEY")
    uses_jev = bool(router_key)
    return {
        "ready": ready,
        "stt": stt_ready,
        "voice": "EMA",
        "voices": 1,
        "device": device_name,
        "agent": bool(read_env("GEMINI_API_KEY")),
        "model": read_env("GEMINI_MODEL") or DEFAULT_MODEL,
        "judge": "jev" if uses_jev else "gemini",
        "judge_model": (read_env("JUDGE_MODEL") or JEV_MODEL) if uses_jev else (read_env("GEMINI_MODEL") or DEFAULT_MODEL),
        "judge_key": mask_key(router_key) if uses_jev else "",
    }


@app.get("/api/token")
def token():
    identity = f"musteri-{uuid.uuid4().hex[:8]}"
    room = f"ema-{uuid.uuid4().hex[:8]}"
    grant = livekit_api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)
    jwt = (
        livekit_api.AccessToken(
            read_env("LIVEKIT_API_KEY") or "devkey",
            read_env("LIVEKIT_API_SECRET") or "secret",
        )
        .with_identity(identity)
        .with_name("Müşteri")
        .with_grants(grant)
        .to_jwt()
    )
    return {"url": read_env("LIVEKIT_URL") or "ws://127.0.0.1:7880", "token": jwt, "room": room}


@app.get("/api/options")
def options():
    return {
        "voices": [
            {
                "id": "ema",
                "name": "EMA",
                "detail": "Tek Türkçe ses. Klonlama ve duygu seçimi yok.",
            }
        ],
        "sample_rates": list(RATES),
        "speed_min": 0.25,
        "speed_max": 4.0,
    }


@app.post("/api/transcribe")
async def transcribe(request: Request):
    if not stt_ready:
        raise HTTPException(status_code=503, detail="Tanıma modeli hazırlanıyor.")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Ses kaydı boş.")
    if len(data) > 15_000_000:
        raise HTTPException(status_code=400, detail="Kayıt çok uzun.")
    try:
        audio, rate = read_wav(data)
    except (wave.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="Ses WAV formatında olmalı.") from error
    audio = to_16k(audio, rate)
    if audio.size < STT_RATE // 5:
        raise HTTPException(status_code=400, detail="Kayıt çok kısa.")
    started = time.perf_counter()
    text = decode_pcm(audio)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return JSONResponse(
        {
            "text": text,
            "audio_seconds": round(audio.size / STT_RATE, 3),
            "stt_ms": round(elapsed_ms, 1),
        }
    )


def synthesize(text: str, speed: float, seed: int, sample_rate: int):
    if sample_rate not in RATES:
        raise HTTPException(status_code=400, detail="Örnekleme 48000, 24000, 16000 veya 8000 olmalı.")
    if isinstance(speed, bool) or not 0.25 <= speed <= 4:
        raise HTTPException(status_code=400, detail="Hız 0.25 ile 4 arasında olmalı.")
    if isinstance(seed, bool) or seed < 0:
        raise HTTPException(status_code=400, detail="Seed sıfır veya pozitif bir tam sayı olmalı.")
    if not ready or tts is None:
        raise HTTPException(status_code=503, detail="Model hazırlanıyor.")
    started = time.perf_counter()
    with lock:
        try:
            speech = tts.say(text, speed=speed, seed=seed, sample_rate=sample_rate)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    elapsed_ms = (time.perf_counter() - started) * 1000
    return speech, elapsed_ms


@app.post("/api/agent")
def agent(body: AgentIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")
    history = [{"role": turn.role, "text": turn.text.strip()} for turn in body.history if turn.text.strip()]
    started = time.perf_counter()
    reply, model = gemini_reply(history, text)
    llm_ms = (time.perf_counter() - started) * 1000
    speech, generate_ms = synthesize(reply, body.speed, body.seed, body.sample_rate)
    return {
        "reply": reply,
        "model": model,
        "llm_ms": round(llm_ms, 1),
        "generate_ms": round(generate_ms, 1),
        "duration": round(speech.duration, 3),
        "sample_rate": speech.sample_rate,
        "seed": speech.seed,
        "audio": base64.b64encode(wav_bytes(speech.audio, speech.sample_rate)).decode(),
    }


@app.post("/api/speak")
def speak(body: SpeakIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")
    speech, elapsed_ms = synthesize(text, body.speed, body.seed, body.sample_rate)
    return Response(
        content=wav_bytes(speech.audio, speech.sample_rate),
        media_type="audio/wav",
        headers={
            "X-Duration": f"{speech.duration:.3f}",
            "X-Seed": str(speech.seed),
            "X-Sample-Rate": str(speech.sample_rate),
            "X-Generate-Ms": f"{elapsed_ms:.1f}",
        },
    )
