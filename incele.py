"""EMA Lightning yetkinlik dinleme seti ve yerel süre ölçümü."""
from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np
import torch
from ema_lightning import EMA

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "samples"
SEED = 0

CASES = [
    ("01_karsilama", "Merhaba, size nasıl yardımcı olabilirim?"),
    (
        "02_odeme",
        "15 Ekim 2026 Çarşamba günü saat 14:30'da, 1.250.000 TL tutarındaki ödemeniz hesabınıza yatırılacak.",
    ),
    (
        "03_liman",
        "Sabahın erken saatlerinde liman henüz uyanmamıştı. Balıkçılar ağlarını sessizce topluyor, martılar ise teknelerin etrafında dönerek şanslarını deniyordu.",
    ),
    ("04_bes_kisi", "5 kişi geldi."),
    ("05_yuzde", "Bugün tüm ürünlerde %15 indirim var."),
    ("06_kilo", "Tarif için 12,5 kg un gerekiyor."),
    ("07_doktor", "Dr. Ayşe geldi."),
    ("08_kod", "Kod: 00042"),
    ("09_tbmm", "TBMM bu hafta toplanacak."),
    ("10_harf", "Parolanız ABC olarak belirlendi."),
    ("11_yabanci", "iPhone ve WhatsApp üzerinden mesaj geldi."),
    ("12_soru", "Yarın saat onda müsait misiniz?"),
    ("13_unlem", "Harika, siparişiniz onaylandı!"),
    ("14_kisa", "Tamam."),
    (
        "15_orta",
        "Siparişiniz yola çıktı. Kargo takip numaranız kısa süre içinde telefonunuza gelecek.",
    ),
]

SPEED_TEXT = "Merhaba, bu cümle farklı hızlarda okunuyor."
SPEEDS = [
    ("16_hiz_075", 0.75),
    ("17_hiz_100", 1.0),
    ("18_hiz_125", 1.25),
]

STREAM_TEXT = (
    "Merhaba! Bu ses siz dinlerken üretiliyor. İlk kelimeyi duyduğunuzda, "
    "cümlenin geri kalanı çoktan hazır."
)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else "CPU"
    lines = [
        f"torch={torch.__version__}",
        f"cuda={cuda}",
        f"device={device_name}",
        f"seed={SEED}",
    ]
    print("\n".join(lines), flush=True)

    t0 = time.perf_counter()
    tts = EMA()
    lines.append(f"load_s={time.perf_counter() - t0:.3f}")
    print(lines[-1], flush=True)

    mode = "plain"
    if cuda:
        try:
            t1 = time.perf_counter()
            tts = tts.lightning()
            lines.append(f"lightning_s={time.perf_counter() - t1:.3f}")
            mode = "lightning"
            print(lines[-1], flush=True)
        except Exception as exc:
            lines.append(f"lightning_failed={type(exc).__name__}: {exc}")
            print(lines[-1], flush=True)
            tts = EMA()
    lines.append(f"mode={mode}")

    print("warmup", flush=True)
    tts.say("Isınma cümlesi.", seed=SEED)
    for _ in tts.stream("Isınma.", seed=SEED):
        break

    rows: list[tuple[str, str, float, float, float, float, int]] = []

    def timed_say(name: str, text: str, **kwargs) -> None:
        t_start = time.perf_counter()
        speech = tts.say(text, seed=SEED, path=str(OUT / f"{name}.wav"), **kwargs)
        elapsed = time.perf_counter() - t_start
        rtf = elapsed / speech.duration if speech.duration else float("inf")
        times = speech.duration / elapsed if elapsed else float("inf")
        rows.append((name, text, speech.duration, elapsed, rtf, times, speech.sample_rate))
        print(
            f"{name}  audio={speech.duration:.2f}s  gen={elapsed:.3f}s  {times:.1f}x",
            flush=True,
        )

    for name, text in CASES:
        timed_say(name, text)
    for name, speed in SPEEDS:
        timed_say(name, SPEED_TEXT, speed=speed)

    print("stream", flush=True)
    t_start = time.perf_counter()
    chunks: list[np.ndarray] = []
    first_s = None
    first_samples = 0
    for chunk in tts.stream(STREAM_TEXT, seed=SEED):
        if first_s is None:
            first_s = time.perf_counter() - t_start
            first_samples = int(chunk.shape[0])
        chunks.append(np.asarray(chunk, dtype=np.float32))
    stream_total = time.perf_counter() - t_start
    streamed = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    said = tts.say(STREAM_TEXT, seed=SEED, path=str(OUT / "19_say.wav"))
    write_wav(OUT / "19_stream.wav", streamed, said.sample_rate)
    same_shape = streamed.shape == said.audio.shape
    max_abs = (
        float(np.max(np.abs(streamed - said.audio))) if same_shape else float("nan")
    )
    # Joined stream audio should match say(); allow tiny numeric noise.
    equals = bool(same_shape and max_abs < 1e-4)

    lines.append("")
    lines.append("name\tduration_s\tgen_s\trtf\tx_realtime\tsample_rate\ttext")
    for name, text, duration, elapsed, rtf, times, sample_rate in rows:
        lines.append(
            f"{name}\t{duration:.3f}\t{elapsed:.3f}\t{rtf:.4f}\t{times:.1f}\t{sample_rate}\t{text}"
        )
    lines.append("")
    lines.append(f"stream_text={STREAM_TEXT}")
    lines.append(f"stream_first_chunk_s={first_s:.4f}" if first_s is not None else "stream_first_chunk_s=nan")
    lines.append(f"stream_first_chunk_samples={first_samples}")
    lines.append(
        f"stream_first_chunk_audio_s={first_samples / said.sample_rate:.3f}"
    )
    lines.append(f"stream_total_s={stream_total:.3f}")
    lines.append(f"say_duration_s={said.duration:.3f}")
    lines.append(f"stream_samples={streamed.shape[0]}")
    lines.append(f"say_samples={said.audio.shape[0]}")
    lines.append(f"stream_equals_say={equals}")
    lines.append(f"stream_say_max_abs_diff={max_abs}")

    report = "\n".join(lines) + "\n"
    (OUT / "olcum.txt").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
