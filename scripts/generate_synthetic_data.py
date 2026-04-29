"""
Fast vectorized synthetic ASVspoof-like dataset generator.
All audio synthesis uses numpy array operations — no per-sample Python loops.

Bonafide: harmonic stack with pitch jitter, formant envelope, naturalamplitude variation, breathiness noise.
Spoof:    stable harmonics (low jitter), over-smooth envelope, near-zero noise, TTS artifacts.
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm

SR       = 16_000
DURATION = 3.0
N        = int(SR * DURATION)   # 48,000 samples

SPLITS = {
    "train":      (2500, 5000),
    "validation": (500,  1000),
    "eval":       (500,  2000),
}

SPOOF_SYSTEMS = [f"A{i:02d}" for i in range(1, 20)]   # A01 … A19

PROTO_NAMES = {
    "train":      "ASVspoof2019.LA.cm.train.trn.txt",
    "validation": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":       "ASVspoof2019.LA.cm.eval.trl.txt",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _formant_filter(signal: np.ndarray, f1: float, f2: float, f3: float) -> np.ndarray:
    """Shape spectrum with three Gaussian formant peaks via FFT."""
    S    = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(N, 1 / SR)
    env  = (np.exp(-((freq - f1) ** 2) / (2 * 200 ** 2))
          + 0.55 * np.exp(-((freq - f2) ** 2) / (2 * 350 ** 2))
          + 0.25 * np.exp(-((freq - f3) ** 2) / (2 * 600 ** 2)))
    env /= env.max() + 1e-9
    return np.fft.irfft(S * env, n=N)


def _amplitude_envelope(rng: np.random.Generator, n_segs: int = 5) -> np.ndarray:
    """Piecewise amplitude envelope with random pauses (prosody)."""
    env = np.ones(N)
    boundaries = np.sort(rng.integers(0, N, size=n_segs - 1))
    segs = np.split(np.arange(N), boundaries)
    for seg in segs:
        if len(seg) == 0:
            continue
        if rng.random() < 0.2:          # pause
            env[seg] *= rng.uniform(0.0, 0.1)
        else:
            env[seg] *= rng.uniform(0.4, 1.0)
    # smooth transitions with a short convolution window
    win = np.hanning(int(0.015 * SR) | 1)
    env = np.convolve(env, win / win.sum(), mode="same")
    return np.clip(env, 0, 1)


# ── Bonafide synthesis ─────────────────────────────────────────────────────────

def generate_bonafide(rng: np.random.Generator) -> np.ndarray:
    """
    Genuine speech simulation:
    - Voiced excitation: stack of harmonics with random amplitude weights
    - Per-harmonic phase jitter (natural micro-variation)
    - Formant envelope from vocal tract filter
    - Natural prosodic amplitude envelope with pauses
    - Breathiness + room noise
    """
    f0      = rng.uniform(80, 300)
    jitter  = rng.uniform(0.004, 0.018)     # pitch jitter per harmonic
    t       = np.arange(N) / SR

    # Harmonic stack with jitter on each partial
    n_harm  = int(SR / (2 * f0))            # harmonics up to Nyquist
    signal  = np.zeros(N, dtype=np.float64)
    for k in range(1, min(n_harm + 1, 30)):
        amp    = rng.uniform(0.3, 1.0) / k ** rng.uniform(0.8, 1.4)
        jit_k  = rng.uniform(-jitter, jitter)
        phase  = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * f0 * k * (1 + jit_k) * t + phase)

    # Formant envelope
    signal = _formant_filter(signal,
                             f1=rng.uniform(500, 900),
                             f2=rng.uniform(1000, 1700),
                             f3=rng.uniform(2200, 2900))

    # Natural prosody
    signal *= _amplitude_envelope(rng, n_segs=rng.integers(4, 8))

    # Noise components
    breathiness = rng.uniform(0.015, 0.07)
    room_noise  = rng.uniform(0.005, 0.025)
    signal     += breathiness * rng.standard_normal(N)
    signal     += room_noise  * rng.standard_normal(N)

    # Normalize
    peak = np.abs(signal).max()
    if peak > 1e-6:
        signal = signal / peak * rng.uniform(0.70, 0.92)
    return signal.astype(np.float32)


# ── Spoof synthesis ────────────────────────────────────────────────────────────

def generate_spoof(rng: np.random.Generator, system_id: str) -> np.ndarray:
    """
    TTS/VC artifact simulation:
    - Very stable harmonics (tiny jitter) — hallmark of neural TTS
    - Fixed amplitude rolloff (over-smooth — no natural variation per partial)
    - Near-zero noise floor
    - Flat amplitude envelope (no prosody)
    - System-specific artifact injection
    """
    f0     = rng.uniform(100, 260)
    jitter = rng.uniform(0.0002, 0.002)     # 5-10x smaller than bonafide
    t      = np.arange(N) / SR

    n_harm = int(SR / (2 * f0))
    signal = np.zeros(N, dtype=np.float64)
    for k in range(1, min(n_harm + 1, 30)):
        amp   = 1.0 / (k ** 1.1)            # unnaturally smooth rolloff
        jit_k = rng.uniform(-jitter, jitter)
        phase = rng.uniform(0, 2 * np.pi) * 0.05   # almost no phase variation
        signal += amp * np.sin(2 * np.pi * f0 * k * (1 + jit_k) * t + phase)

    # Smooth, narrow formant envelope
    signal = _formant_filter(signal,
                             f1=rng.uniform(600, 800),
                             f2=rng.uniform(1100, 1500),
                             f3=rng.uniform(2300, 2700))

    # System-specific artifacts
    sys_idx = int(system_id[1:]) if system_id[1:].isdigit() else 0
    if sys_idx % 4 == 0:
        # Unit-selection splice artifact: random amplitude dips
        for _ in range(rng.integers(2, 6)):
            cut   = rng.integers(0, N - 400)
            width = rng.integers(80, 300)
            signal[cut:cut + width] *= rng.uniform(0.2, 0.6)
    elif sys_idx % 4 == 1:
        # Vocoder periodicity: fine amplitude modulation at frame rate
        frame_rate = rng.uniform(80, 120)   # Hz
        mod = 1.0 + 0.04 * np.sin(2 * np.pi * frame_rate * t)
        signal *= mod
    elif sys_idx % 4 == 2:
        # Neural TTS: very slight global AM at low freq (breath rhythm without breathiness)
        signal *= 1.0 + 0.03 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t)
    # else: clean DNN-TTS, no special artifact

    # Nearly no noise
    signal += rng.uniform(0.0003, 0.003) * rng.standard_normal(N)

    # Flat envelope with clean fade in/out
    fade = int(0.008 * SR)
    signal[:fade]  *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)

    peak = np.abs(signal).max()
    if peak > 1e-6:
        signal = signal / peak * rng.uniform(0.72, 0.93)
    return signal.astype(np.float32)


# ── Dataset builder ────────────────────────────────────────────────────────────

def generate_split(split: str, n_bon: int, n_sp: int,
                   base: Path, rng: np.random.Generator) -> list:
    audio_dir = base / split / "flac"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"\n[{split.upper()}]  bonafide={n_bon}  spoof={n_sp}")

    for i in tqdm(range(n_bon), desc="  bonafide", unit="file"):
        uid   = f"LA_{split[0].upper()}_{i+1:06d}_bonafide"
        fpath = audio_dir / f"{uid}.flac"
        if not fpath.exists():
            sf.write(str(fpath), generate_bonafide(rng), SR)
        rows.append(f"spk{i%50:04d} {uid} - - bonafide")

    for i in tqdm(range(n_sp), desc="  spoof   ", unit="file"):
        sys_id = SPOOF_SYSTEMS[i % len(SPOOF_SYSTEMS)]
        uid    = f"LA_{split[0].upper()}_{i+1:06d}_spoof_{sys_id}"
        fpath  = audio_dir / f"{uid}.flac"
        if not fpath.exists():
            sf.write(str(fpath), generate_spoof(rng, sys_id), SR)
        rows.append(f"spk{(n_bon + i) % 50:04d} {uid} {sys_id} - spoof")

    return rows


def main():
    base         = Path("data/raw/ASVspoof2019_synthetic_LA")
    protocol_dir = base / "ASVspoof2019_LA_cm_protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    proto_paths = {
        "train":      protocol_dir / PROTO_NAMES["train"],
        "validation": protocol_dir / PROTO_NAMES["validation"],
        "eval":       protocol_dir / PROTO_NAMES["eval"],
    }

    for split, (n_bon, n_sp) in SPLITS.items():
        rows = generate_split(split, n_bon, n_sp, base, rng)
        proto_paths[split].write_text("\n".join(rows) + "\n")
        print(f"  Protocol: {proto_paths[split]} ({len(rows)} entries)")

    total = sum(a + b for a, b in SPLITS.values())
    print(f"\nDone! {total} audio files in {base}")


if __name__ == "__main__":
    main()
