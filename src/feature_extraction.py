"""
Feature extraction pipeline for ASVspoof 2019 audio files.
Produces MFCC, LFCC, and log-mel spectrogram representations.
"""

import numpy as np
import librosa
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Optional, Tuple


# ── Defaults aligned with ASVspoof literature ──────────────────────────────
SAMPLE_RATE = 16_000
N_MFCC = 60
N_LFCC = 60
N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160   # 10 ms at 16 kHz
WIN_LENGTH = 400   # 25 ms at 16 kHz


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file and resample to target sample rate. Returns mono float32."""
    waveform, sr = librosa.load(path, sr=target_sr, mono=True)
    return waveform


def extract_mfcc(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    delta: bool = True,
) -> np.ndarray:
    """
    Extract MFCC features with optional delta and delta-delta coefficients.

    Returns: (n_frames, n_mfcc * 3) if delta=True, else (n_frames, n_mfcc)
    """
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )  # shape: (n_mfcc, n_frames)

    if delta:
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate([mfcc, delta1, delta2], axis=0)

    return mfcc.T  # (n_frames, n_coeffs)


def extract_lfcc(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_lfcc: int = N_LFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    delta: bool = True,
) -> np.ndarray:
    """
    Extract LFCC (Linear Frequency Cepstral Coefficients).
    Uses a linear filterbank instead of the mel scale — better suited
    for capturing synthesis artifacts at high frequencies.

    Returns: (n_frames, n_lfcc * 3) if delta=True, else (n_frames, n_lfcc)
    """
    waveform_t = torch.FloatTensor(waveform).unsqueeze(0)  # (1, T)

    lfcc_transform = T.LFCC(
        sample_rate=sr,
        n_lfcc=n_lfcc,
        speckwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
        },
    )
    lfcc = lfcc_transform(waveform_t).squeeze(0).numpy()  # (n_lfcc, n_frames)

    if delta:
        delta1 = librosa.feature.delta(lfcc)
        delta2 = librosa.feature.delta(lfcc, order=2)
        lfcc = np.concatenate([lfcc, delta1, delta2], axis=0)

    return lfcc.T  # (n_frames, n_coeffs)


def extract_log_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    fixed_length: Optional[int] = 300,
) -> np.ndarray:
    """
    Extract log-mel spectrogram for CNN input.
    Pads or truncates to fixed_length frames so batches have uniform shape.

    Returns: (n_mels, fixed_length) — treated as a 2D image by CNNs
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)  # (n_mels, n_frames)

    if fixed_length is not None:
        n_frames = log_mel.shape[1]
        if n_frames < fixed_length:
            pad = fixed_length - n_frames
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant", constant_values=log_mel.min())
        else:
            log_mel = log_mel[:, :fixed_length]

    # Normalize to [0, 1]
    min_val, max_val = log_mel.min(), log_mel.max()
    if max_val > min_val:
        log_mel = (log_mel - min_val) / (max_val - min_val)

    return log_mel.astype(np.float32)


def process_file(
    audio_path: str,
    output_dir: str,
    feature_types: Tuple[str, ...] = ("mfcc", "lfcc", "logmel"),
    sr: int = SAMPLE_RATE,
) -> None:
    """
    Extract and save all requested feature types for a single audio file.
    Output files: {utterance_id}_{feature_type}.npy
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    utterance_id = Path(audio_path).stem
    waveform = load_audio(audio_path, target_sr=sr)

    extractors = {
        "mfcc":   lambda w: extract_mfcc(w, sr=sr),
        "lfcc":   lambda w: extract_lfcc(w, sr=sr),
        "logmel": lambda w: extract_log_mel_spectrogram(w, sr=sr),
    }

    for feat_type in feature_types:
        out_path = output_dir / f"{utterance_id}_{feat_type}.npy"
        if out_path.exists():
            continue
        features = extractors[feat_type](waveform)
        np.save(str(out_path), features)
