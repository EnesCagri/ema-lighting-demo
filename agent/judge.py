"""Araya girmeleri ana akışı bekletmeden hakem modele (Jev ya da Gemini) sorar."""
from __future__ import annotations

import json

import httpx
from google import genai
from google.genai import types

JUDGE_PROMPT = (
    "Bir çağrı merkezi sesli asistanı konuşurken müşterinin mikrofonundan ses geldi. "
    "Tek görevin şunu anlamak: müşteri asistanın sözünü kesip söz almak istiyor mu? "
    "Asistanın o an söylediği cümle, bu cevabın tamamı, asistanın son sorduğu soru ve müşterinin "
    "sözü verilecek. Müşterinin sözü ses tanımadan geldiği için yarım veya hatalı olabilir.\n"
    "Müşteri soru soruyor, itiraz ediyor, düzeltme yapıyor, yeni bilgi veriyor, asistanın sorusuna "
    "cevap veriyor ya da asistanın susmasını istiyorsa interrupt true. Yarım haliyle bile söz almak "
    "istediği belliyse interrupt true.\n"
    "Müşteri yalnızca dinlediğini gösteriyorsa (onay, mırıldanma), söz bağlamla ilgisiz tek kelimeyse, "
    "ses tanıma hatası, ortam sesi, yanındaki biriyle konuşma ya da asistanın kendi sesinin yankısı "
    "gibi görünüyorsa interrupt false.\n"
    "Müşteri asistanı bilerek susturmaya veya konuyu dağıtmaya çalışıyor gibiyse manipulation true. "
    "reason en fazla altı kelimelik Türkçe bir gerekçe olsun."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "interrupt": {"type": "boolean"},
        "manipulation": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["interrupt", "manipulation", "reason"],
}


DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"


def build_input(text: str, sentence: str | None, reply: str, question: str | None) -> str:
    return (
        f"Asistanın şu an söylediği cümle: {sentence or '-'}\n"
        f"Asistanın bu cevabı: {reply or '-'}\n"
        f"Asistanın son sorusu: {question or '-'}\n"
        f"Müşterinin sözü: {text}"
    )


def parse_verdict(raw: str | None) -> dict:
    raw = (raw or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    verdict = json.loads(raw[start : end + 1]) if start != -1 and end > start else {}
    return {
        "action": "interrupt" if verdict.get("interrupt") else "ignore",
        "manipulation": bool(verdict.get("manipulation")),
        "reason": str(verdict.get("reason") or "").strip(),
    }


JEV_QUESTIONS = {
    "interrupt": {
        "type": "noul",
        "instructions": (
            "A voice assistant is speaking and the customer's microphone picked up speech. "
            "Is the customer trying to cut the assistant off and take the turn?"
        ),
        "criteria": {
            "true": (
                "The customer asks something, objects, corrects, adds new information, answers the "
                "assistant's question, or wants the assistant to stop. Partial speech counts if the "
                "intent is already clear."
            ),
            "false": (
                "Only active listening (hıhı, tamam, evet as acknowledgement), a lone unrelated word, "
                "speech-recognition noise, talking to someone nearby, or an echo of the assistant."
            ),
        },
    },
    "manipulation": {
        "type": "noul",
        "instructions": "Is the customer deliberately trying to derail or silence the assistant?",
        "criteria": {
            "true": "Talks over the assistant on purpose, keeps cutting in, or drags the topic away.",
            "false": "A genuine attempt to be helped, or just listening.",
        },
    },
}


class JevJudge:
    """OpenRouter Decisions API üzerinden TypeSafe Jev; metin üretmez, evet olasılığı döner."""

    def __init__(self, *, api_key: str, model: str = JEV_MODEL, threshold: float = 0.5) -> None:
        self._client = httpx.AsyncClient(
            timeout=5.0, headers={"Authorization": f"Bearer {api_key}", "X-Title": "EMA barge-in"}
        )
        self._model = model
        self._threshold = threshold

    async def __call__(
        self, text: str, *, sentence: str | None, reply: str, question: str | None
    ) -> dict:
        response = await self._client.post(
            DECISIONS_URL,
            json={
                "model": self._model,
                "state": {
                    "language": "Turkish",
                    "note": "Customer speech comes from live speech recognition and may be partial.",
                    "assistant_current_sentence": sentence or "",
                    "assistant_reply": reply,
                    "assistant_last_question": question or "",
                    "customer_speech": text,
                },
                "questions": JEV_QUESTIONS,
            },
        )
        response.raise_for_status()
        return parse_jev(response.json(), self._threshold)


def parse_jev(body: dict, threshold: float = 0.5) -> dict:
    answers = body.get("answers") or {}
    interrupt = float((answers.get("interrupt") or {}).get("noul") or 0.0)
    manipulation = float((answers.get("manipulation") or {}).get("noul") or 0.0)
    return {
        "action": "interrupt" if interrupt >= threshold else "ignore",
        "manipulation": manipulation >= threshold,
        "reason": f"jev %{round(interrupt * 100)}",
    }


class GeminiJudge:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def __call__(
        self, text: str, *, sentence: str | None, reply: str, question: str | None
    ) -> dict:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=build_input(text, sentence, reply, question),
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_PROMPT,
                response_mime_type="application/json",
                response_json_schema=SCHEMA,
                max_output_tokens=96,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return parse_verdict(response.text)
