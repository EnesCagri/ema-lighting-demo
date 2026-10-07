from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gemini-3.1-flash-lite"
AGENT_PROMPT = (
    "Sen Migros Sanal Market çağrı merkezinin sesli asistanısın. "
    "Karşındaki kişi telefonda konuşuyor. "
    "Yalnızca sesli okunacak cevabı yaz. Madde işareti, emoji ve markdown kullanma. "
    "En fazla iki kısa cümle söyle. Türkçe konuş. "
    "Sayıları ve tarihleri konuşma dilinde yaz. "
    "Bilmediğin sipariş, tutar veya adres uydurma. Eksik bilgiyi tek soruyla sor."
)


def read_env(name: str) -> str:
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return os.environ.get(name, "").strip()
