"""EMA konuşurken araya giren sözün kesip kesmeyeceğine karar verir."""
from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

BACKCHANNEL = {
    "hı", "hıhı", "hıı", "ıhı", "ıı", "ı", "hm", "hmm", "mhm", "mm", "he", "hee", "ha", "haa",
    "evet", "tamam", "anladım", "anlıyorum", "peki", "aynen", "doğru", "olur", "tabii", "tabi",
    "iyi", "güzel", "harika", "süper", "ok", "okey",
}
ANSWERS = {
    "evet", "hayır", "tamam", "olur", "yok", "var", "doğru", "değil", "istiyorum", "istemiyorum",
    "isterim", "istemem", "aynen", "tabii", "tabi", "olmaz", "uygun",
}
STOP_PHRASES = (
    "dur", "durun", "bekle", "bekleyin", "bir saniye", "bi saniye", "bir dakika", "bi dakika",
    "hayır hayır", "yanlış", "öyle değil", "kes", "susun", "sus",
)
CONTINUATION = {"ve", "ama", "şey", "yani", "çünkü", "ee", "eee", "ıı", "fakat", "veya", "ki", "ile", "sonra"}

MEANINGFUL_WORDS = 3
MEANINGFUL_SECONDS = 1.2
SHORT_NOISE_SECONDS = 0.6
JUDGED_REASONS = {"anlamlı söz", "kısa ses", "yankı", "belirsiz"}
MAX_JUDGE_CALLS = 2

Action = Literal["interrupt", "ignore", "judge"]


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str


def normalize(text: str) -> str:
    text = text.replace("I", "ı").replace("İ", "i").lower()
    text = re.sub(r"[^\wçğıöşü\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return normalize(text).split()


def is_backchannel(text: str) -> bool:
    tokens = words(text)
    return bool(tokens) and all(token in BACKCHANNEL for token in tokens)


def ends_with_continuation(text: str) -> bool:
    tokens = words(text)
    return bool(tokens) and tokens[-1] in CONTINUATION


def has_stop_phrase(text: str) -> bool:
    padded = f" {normalize(text)} "
    return any(f" {phrase} " in padded for phrase in STOP_PHRASES)


def is_question(sentence: str | None) -> bool:
    return bool(sentence) and sentence.rstrip().endswith("?")


def is_echo(text: str, agent_text: str) -> bool:
    """Araya giren kelimelerin hepsi EMA'nın o an söylediği cümlede geçiyorsa bu kendi sesinin yankısıdır."""
    tokens = words(text)
    spoken = words(agent_text)
    if not tokens or not spoken:
        return False
    stems = {token[:3] for token in spoken if len(token) >= 3}
    return all(token in spoken or (len(token) >= 3 and token[:3] in stems) for token in tokens)


def classify(
    text: str, *, question: bool, speech_seconds: float, agent_text: str = ""
) -> Decision:
    tokens = words(text)
    if not tokens:
        return Decision("ignore", "boş")
    if has_stop_phrase(text):
        return Decision("interrupt", "durdurma")
    if question and (tokens[0] in ANSWERS or all(token in ANSWERS for token in tokens)):
        return Decision("interrupt", "soruya cevap")
    if all(token in BACKCHANNEL for token in tokens):
        return Decision("ignore", "onay")
    if len(tokens) <= 2 and is_echo(text, agent_text):
        return Decision("ignore", "yankı")
    meaningful = [token for token in tokens if token not in BACKCHANNEL]
    if len(meaningful) >= MEANINGFUL_WORDS or speech_seconds >= MEANINGFUL_SECONDS:
        return Decision("interrupt", "anlamlı söz")
    if len(meaningful) == 1 and speech_seconds < SHORT_NOISE_SECONDS:
        return Decision("ignore", "kısa ses")
    return Decision("judge", "belirsiz")


class OverTalkGuard:
    """Sürekli üstüne konuşmayı sayar; eşik aşılınca kesmeyi durdurur ve uyarı ister."""

    def __init__(
        self,
        *,
        window: float = 30.0,
        max_interrupts: int = 3,
        max_overlap: float = 6.0,
        cooldown: float = 60.0,
    ) -> None:
        self.window = window
        self.max_interrupts = max_interrupts
        self.max_overlap = max_overlap
        self.cooldown = cooldown
        self._events: deque[float] = deque()
        self._overlap_since: float | None = None
        self._last_warning: float | None = None

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0] > self.window:
            self._events.popleft()

    def record(self, now: float) -> None:
        self._events.append(now)
        self._trim(now)

    def overlap(self, active: bool, now: float) -> None:
        if active and self._overlap_since is None:
            self._overlap_since = now
        elif not active:
            self._overlap_since = None

    def overlap_seconds(self, now: float) -> float:
        return 0.0 if self._overlap_since is None else now - self._overlap_since

    def blocked(self, now: float) -> bool:
        self._trim(now)
        return len(self._events) >= self.max_interrupts or self.overlap_seconds(now) >= self.max_overlap

    def should_warn(self, now: float) -> bool:
        if not self.blocked(now):
            return False
        return self._last_warning is None or now - self._last_warning >= self.cooldown

    def warned(self, now: float) -> None:
        self._last_warning = now


JudgeFn = Callable[..., Awaitable[dict]]


class BargeInGate:
    """LiveKit'ten bağımsız karar kapısı. Olaylar worker'dan gelir, eylemler geri çağrılarla çıkar."""

    def __init__(
        self,
        *,
        interrupt: Callable[[], None],
        warn: Callable[[float], None],
        emit: Callable[[dict], None] = lambda _event: None,
        judge: JudgeFn | None = None,
        judge_timeout: float = 1.5,
        judge_first: bool = True,
        guard: OverTalkGuard | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._interrupt = interrupt
        self._warn = warn
        self._emit = emit
        self._judge = judge
        self._judge_timeout = judge_timeout
        self._judged_reasons = JUDGED_REASONS if judge_first else {"belirsiz"}
        self.guard = guard or OverTalkGuard()
        self._clock = clock
        self._timeline: list[tuple[str, float]] = []
        self._speaking_since: float | None = None
        self._user_since: float | None = None
        self._overlap_id = 0
        self._handled = False
        self._judge_calls = 0
        self._judged_text = ""
        self._pending: tuple[str, Decision] | None = None
        self._overtalk_counted = False
        self._last_question: str | None = None
        self._last_reply = ""
        self._drop_backchannel = False
        self._held = ""
        self._resumed = asyncio.Event()
        self.tasks: set[asyncio.Task] = set()

    @property
    def agent_speaking(self) -> bool:
        return self._speaking_since is not None

    def add_sentence(self, text: str, duration: float) -> None:
        self._timeline.append((text, duration))
        if is_question(text):
            self._last_question = text

    def set_agent_speaking(self, speaking: bool) -> None:
        now = self._clock()
        if speaking and self._speaking_since is None:
            self._speaking_since = now
            if self._user_since is not None:
                self._start_overlap(now)
        elif not speaking and self._speaking_since is not None:
            self._speaking_since = None
            if self._timeline:
                self._last_reply = " ".join(text for text, _ in self._timeline)
            self._timeline = []
            self.guard.overlap(False, now)

    @property
    def last_question(self) -> str | None:
        return self._last_question

    @property
    def last_reply(self) -> str:
        return self._last_reply

    def sentences(self) -> tuple[str | None, str | None, float]:
        """Çalan cümle, ondan önceki cümle ve çalan cümlenin bitmesine kalan süre."""
        if self._speaking_since is None or not self._timeline:
            return None, None, 0.0
        elapsed = self._clock() - self._speaking_since
        start = 0.0
        previous = None
        for text, duration in self._timeline:
            if elapsed < start + duration:
                return text, previous, start + duration - elapsed
            previous = text
            start += duration
        return previous, None, 0.0

    def _start_overlap(self, now: float) -> None:
        self._overlap_id += 1
        self._handled = False
        self._judge_calls = 0
        self._judged_text = ""
        self._pending = None
        self._overtalk_counted = False
        self.guard.overlap(True, now)

    def set_user_speaking(self, speaking: bool) -> None:
        now = self._clock()
        if speaking:
            self._user_since = now
            self._resumed.set()
            if self.agent_speaking:
                self._start_overlap(now)
            else:
                self._drop_backchannel = False
        else:
            self._user_since = None
            self.guard.overlap(False, now)

    def on_interim(self, text: str) -> Decision | None:
        if not self.agent_speaking or self._handled:
            return None
        now = self._clock()
        seconds = 0.0 if self._user_since is None else now - self._user_since
        current, previous, _ = self.sentences()
        decision = classify(
            text,
            question=is_question(current) or is_question(previous),
            speech_seconds=seconds,
            agent_text=" ".join(part for part in (previous, current) if part),
        )

        if self._judge is not None and decision.reason in self._judged_reasons:
            self._request_judge(text, decision)
            decision = Decision("judge", decision.reason)
        elif decision.action == "interrupt":
            self._try_interrupt(text, decision.reason)
        else:
            if decision.reason == "onay":
                self._drop_backchannel = True
            self._emit(
                {"type": "gate", "action": "ignore", "via": "kural", "reason": decision.reason, "text": text}
            )

        if self.guard.overlap_seconds(now) >= self.guard.max_overlap:
            self._maybe_warn(now)
        return decision

    def _request_judge(self, text: str, fallback: Decision) -> None:
        """Her yeni ara metni hakeme sorar; en fazla iki çağrı aynı anda açık kalır, fazlası en yenisiyle bekler."""
        if text == self._judged_text:
            return
        if self._judge_calls >= MAX_JUDGE_CALLS:
            self._pending = (text, fallback)
            return
        self._judged_text = text
        self._judge_calls += 1
        self._emit(
            {"type": "gate", "action": "judge", "via": "jev", "reason": fallback.reason, "text": text}
        )
        task = asyncio.ensure_future(self._ask_judge(text, fallback, self._overlap_id))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def _try_interrupt(self, text: str, reason: str, extra: dict | None = None) -> None:
        now = self._clock()
        self._handled = True
        extra = extra or {}
        if self.guard.blocked(now):
            self._emit(
                {"type": "gate", "action": "blocked", "via": extra.get("via", "kural"), "reason": reason, "text": text, **extra}
            )
            self._maybe_warn(now)
            return
        self.guard.record(now)
        self._drop_backchannel = False
        self._emit(
            {"type": "gate", "action": "interrupt", "via": extra.get("via", "kural"), "reason": reason, "text": text, **extra}
        )
        self._interrupt()

    def _maybe_warn(self, now: float) -> None:
        if not self.guard.should_warn(now):
            return
        self.guard.warned(now)
        _, _, remaining = self.sentences()
        self._emit({"type": "gate", "action": "warn", "via": "kural", "reason": "üstüne konuşma", "text": ""})
        self._warn(remaining)

    async def _ask_judge(self, text: str, fallback: Decision, overlap_id: int) -> None:
        assert self._judge is not None
        current, _, _ = self.sentences()
        reply = " ".join(sentence for sentence, _ in self._timeline)
        try:
            verdict: dict | None = await asyncio.wait_for(
                self._judge(text, sentence=current, reply=reply, question=self._last_question),
                self._judge_timeout,
            )
        except Exception:
            verdict = None
        if overlap_id != self._overlap_id:
            return
        self._judge_calls -= 1
        if verdict is not None and verdict.get("manipulation") and not self._overtalk_counted:
            self._overtalk_counted = True
            self.guard.record(self._clock())
        if not self.agent_speaking or self._handled:
            return

        if verdict is None:
            if fallback.action == "interrupt":
                self._try_interrupt(text, f"{fallback.reason}, hakem yetişmedi")
            else:
                self._emit(
                    {
                        "type": "gate",
                        "action": "ignore",
                        "via": "kural",
                        "reason": "hakem yetişmedi",
                        "text": text,
                    }
                )
        else:
            reason = verdict.get("reason") or "hakem"
            extra = {"by": "hakem", "via": "jev"}
            if verdict.get("action") == "interrupt":
                self._try_interrupt(text, reason, extra)
            else:
                self._emit({"type": "gate", "action": "ignore", "reason": reason, "text": text, **extra})

        if self._pending is not None and not self._handled:
            pending_text, pending_fallback = self._pending
            self._pending = None
            self._request_judge(pending_text, pending_fallback)

    def should_drop_turn(self, text: str) -> bool:
        drop = self._drop_backchannel and is_backchannel(text)
        self._drop_backchannel = False
        return drop

    def merge_held(self, text: str) -> str:
        held, self._held = self._held, ""
        return f"{held} {text}".strip() if held else text

    def hold(self, text: str) -> None:
        self._held = text

    async def wait_user_resume(self, timeout: float) -> bool:
        if self._user_since is not None:
            return True
        self._resumed.clear()
        try:
            await asyncio.wait_for(self._resumed.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False
