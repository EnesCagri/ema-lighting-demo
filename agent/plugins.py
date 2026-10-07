"""LiveKit eklentileri: yerel Zipformer dinler, EMA konuşur."""
from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sherpa_onnx
from huggingface_hub import snapshot_download
from livekit.agents import APIConnectOptions, stt, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

STT_RATE = 16_000
TTS_RATE = 48_000
STT_ID = "duxx/turkish-stt-zipformer"
STT_VARIANT = "epoch-1-avg-1-chunk-32-left-128"
STT_FILES = {
    "tokens": "tokens.txt",
    "encoder": f"encoder-{STT_VARIANT}.int8.onnx",
    "decoder": f"decoder-{STT_VARIANT}.int8.onnx",
    "joiner": f"joiner-{STT_VARIANT}.int8.onnx",
}

_load_lock = threading.Lock()
_stt_lock = threading.Lock()
_tts_lock = threading.Lock()
_recognizer = None
_ema = None


def load_recognizer(*, endpoints: bool = True):
    global _recognizer
    with _load_lock:
        if _recognizer is None:
            folder = Path(snapshot_download(STT_ID, allow_patterns=list(STT_FILES.values())))
            _recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(folder / STT_FILES["tokens"]),
                encoder=str(folder / STT_FILES["encoder"]),
                decoder=str(folder / STT_FILES["decoder"]),
                joiner=str(folder / STT_FILES["joiner"]),
                num_threads=min(4, os.cpu_count() or 2),
                decoding_method="modified_beam_search",
                max_active_paths=12,
                blank_penalty=0.0,
                provider="cpu",
                enable_endpoint_detection=endpoints,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=0.25,
                rule3_min_utterance_length=20.0,
            )
        return _recognizer


def load_ema():
    global _ema
    with _load_lock:
        if _ema is None:
            from ema_lightning import EMA

            _ema = EMA()
            _ema.say("Merhaba.", seed=0)
        return _ema


def result_text(recognizer, stream) -> str:
    result = recognizer.get_result(stream)
    text = result if isinstance(result, str) else result.text
    return text.strip().lower()


def decode_pcm(audio: np.ndarray) -> str:
    """Bütün bir kaydı yazıya döker (endpoint kuralları bunu bölmez)."""
    recognizer = load_recognizer()
    with _stt_lock:
        stream = recognizer.create_stream()
        stream.accept_waveform(STT_RATE, audio)
        stream.accept_waveform(STT_RATE, np.zeros(STT_RATE, dtype=np.float32))
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        return result_text(recognizer, stream)


def frame_to_float(data: bytes | memoryview) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


class ZipformerSTT(stt.STT):
    def __init__(self, language: str = "tr") -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True)
        )
        self._language = language
        self._recognizer = load_recognizer()

    @property
    def model(self) -> str:
        return STT_ID

    @property
    def provider(self) -> str:
        return "sherpa-onnx"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        frame = utils.merge_frames(buffer)
        audio = frame_to_float(frame.data)
        if frame.sample_rate != STT_RATE:
            source = np.arange(audio.size) / frame.sample_rate
            target = np.arange(int(audio.size * STT_RATE / frame.sample_rate)) / STT_RATE
            audio = np.interp(target, source, audio).astype(np.float32)
        text = await asyncio.to_thread(decode_pcm, audio)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=self._language, text=text, confidence=1.0)],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ZipformerStream:
        return ZipformerStream(stt=self, conn_options=conn_options, language=self._language)


class ZipformerStream(stt.RecognizeStream):
    def __init__(self, *, stt: ZipformerSTT, conn_options: APIConnectOptions, language: str) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=STT_RATE)
        self._recognizer = stt._recognizer
        self._language = language

    def _feed(self, stream, audio: np.ndarray) -> tuple[str, bool]:
        with _stt_lock:
            stream.accept_waveform(STT_RATE, audio)
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
            return result_text(self._recognizer, stream), self._recognizer.is_endpoint(stream)

    def _send(self, kind: stt.SpeechEventType, text: str = "") -> None:
        alternatives = (
            [stt.SpeechData(language=self._language, text=text, confidence=1.0)] if text else []
        )
        self._event_ch.send_nowait(stt.SpeechEvent(type=kind, alternatives=alternatives))

    async def _run(self) -> None:
        stream = self._recognizer.create_stream()
        pending: list[np.ndarray] = []
        pending_samples = 0
        last = ""
        speaking = False

        def finish(text: str) -> None:
            nonlocal last, speaking, stream
            if text:
                self._send(stt.SpeechEventType.FINAL_TRANSCRIPT, text)
            if speaking:
                self._send(stt.SpeechEventType.END_OF_SPEECH)
            with _stt_lock:
                self._recognizer.reset(stream)
            last = ""
            speaking = False

        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                if pending:
                    text, _ = await asyncio.to_thread(self._feed, stream, np.concatenate(pending))
                    pending, pending_samples = [], 0
                    last = text or last
                finish(last)
                continue

            chunk = frame_to_float(item.data)
            pending.append(chunk)
            pending_samples += chunk.size
            if pending_samples < STT_RATE // 10:
                continue
            audio = np.concatenate(pending)
            pending, pending_samples = [], 0
            text, endpoint = await asyncio.to_thread(self._feed, stream, audio)

            if text and not speaking:
                speaking = True
                self._send(stt.SpeechEventType.START_OF_SPEECH)
            if text and text != last:
                last = text
                self._send(stt.SpeechEventType.INTERIM_TRANSCRIPT, text)
            if endpoint:
                finish(text)


SentenceHook = Callable[[str, float], None]


class EmaTTS(tts.TTS):
    def __init__(self, *, speed: float = 1.0, seed: int = 0) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=TTS_RATE,
            num_channels=1,
        )
        self.speed = speed
        self.seed = seed
        self._ema = load_ema()
        self._hooks: list[SentenceHook] = []

    @property
    def model(self) -> str:
        return "ema-lightning"

    @property
    def provider(self) -> str:
        return "local"

    def on_sentence(self, hook: SentenceHook) -> None:
        self._hooks.append(hook)

    def _say(self, text: str) -> tuple[bytes, float]:
        with _tts_lock:
            speech = self._ema.say(text, speed=self.speed, seed=self.seed, sample_rate=TTS_RATE)
        pcm = (np.clip(speech.audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
        return pcm.tobytes(), float(speech.duration)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> EmaStream:
        return EmaStream(tts=self, input_text=text, conn_options=conn_options)


class EmaStream(tts.ChunkedStream):
    def __init__(self, *, tts: EmaTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._ema_tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=TTS_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )
        text = self.input_text.strip()
        if not text:
            output_emitter.flush()
            return
        pcm, duration = await asyncio.to_thread(self._ema_tts._say, text)
        for hook in self._ema_tts._hooks:
            hook(text, duration)
        output_emitter.push(pcm)
        output_emitter.flush()
