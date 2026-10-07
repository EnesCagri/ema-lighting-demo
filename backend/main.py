"""EMA Lightning konuşma servisi. Tek model, tek ses."""
from __future__ import annotations

import io
import threading
import time
import wave
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ema_lightning import EMA

RATES = (48000, 24000, 16000, 8000)
tts: EMA | None = None
device_name = "yükleniyor"
ready = False
lock = threading.Lock()


def wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global tts, device_name, ready
    import torch

    tts = EMA()
    device_name = torch.cuda.get_device_name(0) if tts.device.type == "cuda" else "CPU"
    tts.say("Merhaba.", seed=0)
    ready = True
    yield


app = FastAPI(title="EMA Lightning", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Duration", "X-Seed", "X-Sample-Rate", "X-Generate-Ms"],
)


class SpeakIn(BaseModel):
    text: str = Field(max_length=8000)
    speed: float = 1.0
    seed: int = 0
    sample_rate: int = 48000


@app.get("/api/health")
def health():
    return {
        "ready": ready,
        "voice": "EMA",
        "voices": 1,
        "device": device_name,
    }


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


@app.post("/api/speak")
def speak(body: SpeakIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")
    if body.sample_rate not in RATES:
        raise HTTPException(status_code=400, detail="Örnekleme 48000, 24000, 16000 veya 8000 olmalı.")
    if isinstance(body.speed, bool) or not 0.25 <= body.speed <= 4:
        raise HTTPException(status_code=400, detail="Hız 0.25 ile 4 arasında olmalı.")
    if isinstance(body.seed, bool) or body.seed < 0:
        raise HTTPException(status_code=400, detail="Seed sıfır veya pozitif bir tam sayı olmalı.")
    if not ready or tts is None:
        raise HTTPException(status_code=503, detail="Model hazırlanıyor.")

    started = time.perf_counter()
    with lock:
        try:
            speech = tts.say(text, speed=body.speed, seed=body.seed, sample_rate=body.sample_rate)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    elapsed_ms = (time.perf_counter() - started) * 1000
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
