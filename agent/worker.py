"""LiveKit sesli ajanı: Zipformer dinler, Gemini cevaplar, EMA konuşur, BargeInGate araya girmeyi yönetir."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from livekit.agents import (  # noqa: E402
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
)
from livekit.agents.llm import ChatContext, ChatMessage, StopResponse  # noqa: E402
from livekit.plugins import google  # noqa: E402

from agent.bargein import BargeInGate, ends_with_continuation  # noqa: E402
from agent.config import AGENT_PROMPT, DEFAULT_MODEL, read_env  # noqa: E402
from agent.judge import JEV_MODEL, GeminiJudge, JevJudge  # noqa: E402
from agent.plugins import EmaTTS, ZipformerSTT, load_ema, load_recognizer  # noqa: E402

logger = logging.getLogger("ema-agent")

WARNING = "Sizi dinliyorum. Bilgiyi tamamlamama izin verirseniz daha hızlı yardımcı olabilirim."
CONTINUATION_WAIT = 1.2
AWAY_SECONDS = 8.0
MAX_NUDGES = 2

os.environ.setdefault("LIVEKIT_URL", read_env("LIVEKIT_URL") or "ws://127.0.0.1:7880")
os.environ.setdefault("LIVEKIT_API_KEY", read_env("LIVEKIT_API_KEY") or "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", read_env("LIVEKIT_API_SECRET") or "secret")


class EmaAgent(Agent):
    def __init__(self, gate: BargeInGate) -> None:
        super().__init__(instructions=AGENT_PROMPT)
        self.gate = gate

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        text = (new_message.text_content or "").strip()
        if not text or self.gate.should_drop_turn(text):
            raise StopResponse()
        text = self.gate.merge_held(text)
        new_message.content = [text]
        if ends_with_continuation(text) and await self.gate.wait_user_resume(CONTINUATION_WAIT):
            self.gate.hold(text)
            raise StopResponse()


def prewarm(_proc: JobProcess) -> None:
    load_recognizer()
    load_ema()


server = AgentServer(
    setup_fnc=prewarm,
    num_idle_processes=1,
    initialize_process_timeout=180.0,
    load_threshold=0.95,
)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    key = read_env("GEMINI_API_KEY")
    model = read_env("GEMINI_MODEL") or DEFAULT_MODEL
    room = ctx.room
    session: AgentSession | None = None

    def emit(event: dict) -> None:
        logger.info("gate %s", event)
        asyncio.ensure_future(
            room.local_participant.publish_data(
                json.dumps(event, ensure_ascii=False), topic="gate", reliable=True
            )
        )

    def interrupt() -> None:
        if session is not None:
            session.interrupt(force=True)

    async def warn_after(remaining: float) -> None:
        await asyncio.sleep(remaining)
        if session is None:
            return
        session.interrupt(force=True)
        await session.say(WARNING, allow_interruptions=False)
        session.generate_reply(instructions="Yarım kalan bilgiyi bir iki kısa cümleyle tamamla.")

    def warn(remaining: float) -> None:
        asyncio.ensure_future(warn_after(remaining))

    router_key = read_env("OPENROUTER_API_KEY")
    judge_timeout = 1.5
    if router_key:
        judge_model = read_env("JUDGE_MODEL") or JEV_MODEL
        judge = JevJudge(api_key=router_key, model=judge_model)
        judge_timeout = 0.8
        logger.info("hakem: %s", judge_model)
    elif key:
        judge = GeminiJudge(api_key=key, model=model)
        logger.info("hakem: Gemini %s", model)
    else:
        judge = None

    gate = BargeInGate(
        interrupt=interrupt,
        warn=warn,
        emit=emit,
        judge=judge,
        judge_timeout=judge_timeout,
        judge_first=read_env("BARGEIN_MODE") != "rules",
    )
    tts = EmaTTS()
    tts.on_sentence(gate.add_sentence)

    session = AgentSession(
        stt=ZipformerSTT(),
        llm=google.LLM(
            model=model,
            api_key=key,
            max_output_tokens=256,
            thinking_config={"thinking_level": "minimal"},
        ),
        tts=tts,
        vad=inference.VAD(min_silence_duration=0.25),
        aec_warmup_duration=None,
        user_away_timeout=AWAY_SECONDS,
        turn_handling={
            "turn_detection": inference.TurnDetector(version="v1-mini"),
            "endpointing": {"min_delay": 0.2, "max_delay": 1.2},
            "interruption": {"enabled": False, "discard_audio_if_uninterruptible": False},
        },
    )
    nudges = 0

    @session.on("agent_state_changed")
    def _agent_state(ev) -> None:
        gate.set_agent_speaking(ev.new_state == "speaking")

    async def nudge() -> None:
        nonlocal nudges
        if session is None or nudges >= MAX_NUDGES:
            return
        nudges += 1
        if nudges == 1 and gate.last_reply:
            text = gate.last_reply
        elif nudges == 1:
            text = "Sizi dinliyorum. Buyurun, size nasıl yardımcı olabilirim?"
        else:
            text = "Orada mısınız? " + (gate.last_question or "Cevabınızı bekliyorum.")
        emit({"type": "gate", "action": "nudge", "via": "kural", "reason": "sessizlik", "text": text})
        await session.say(text, allow_interruptions=True)

    @session.on("user_state_changed")
    def _user_state(ev) -> None:
        nonlocal nudges
        gate.set_user_speaking(ev.new_state == "speaking")
        if ev.new_state == "speaking":
            nudges = 0
        elif ev.new_state == "away":
            asyncio.ensure_future(nudge())

    @session.on("user_input_transcribed")
    def _transcribed(ev) -> None:
        if not ev.transcript:
            return
        decision = gate.on_interim(ev.transcript)
        if ev.is_final and decision is None and not gate.agent_speaking:
            emit(
                {
                    "type": "gate",
                    "action": "pass",
                    "via": "kural",
                    "reason": "EMA susuyordu",
                    "text": ev.transcript,
                }
            )

    await session.start(agent=EmaAgent(gate), room=room)
    await session.say("Migros Sanal Market'e hoş geldiniz. Size nasıl yardımcı olabilirim?")


if __name__ == "__main__":
    cli.run_app(server)
