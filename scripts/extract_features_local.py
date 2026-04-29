"""
Feature extraction for locally stored ASVspoof-format audio.
Reads protocol files to get utterance IDs + labels, then extracts MFCC/LFCC/log-mel.

Usage:
    python scripts/extract_features_local.py \
        --data_dir data/raw/ASVspoof2019_synthetic_LA \
        --output_dir data/features
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm
from src.feature_extraction import (
    load_audio, extract_mfcc, extract_lfcc, extract_log_mel_spectrogram, SAMPLE_RATE
)
from src.data_loader import parse_protocol


SPLIT_PROTO = {
    "train":      "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "validation": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":       "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}

SPLIT_AUDIO = {
    "train":      "train/flac",
    "validation": "validation/flac",
    "eval":       "eval/flac",
}

EXTRACTORS = {
    "mfcc":   lambda w: extract_mfcc(w),
    "lfcc":   lambda w: extract_lfcc(w),
    "logmel": lambda w: extract_log_mel_spectrogram(w),
}


def process_split(data_dir: Path, output_dir: Path, split: str, feature_types: list):
    proto_path = data_dir / SPLIT_PROTO[split]
    audio_dir  = data_dir / SPLIT_AUDIO[split]
    out_dir    = output_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_protocol(str(proto_path))
    print(f"\n[{split.upper()}] {len(df)} samples -> {out_dir}")

    skipped, failed = 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
        utt_id = row["utterance_id"]
        audio_path = audio_dir / f"{utt_id}.flac"

        if not audio_path.exists():
            failed += 1
            continue

        # Check if all features already extracted
        all_exist = all((out_dir / f"{utt_id}_{ft}.npy").exists() for ft in feature_types)
        if all_exist:
            skipped += 1
            continue

        try:
            waveform = load_audio(str(audio_path))
        except Exception as e:
            print(f"  Load error {utt_id}: {e}")
            failed += 1
            continue

        for ft in feature_types:
            out_path = out_dir / f"{utt_id}_{ft}.npy"
            if out_path.exists():
                continue
            try:
                feat = EXTRACTORS[ft](waveform)
                np.save(str(out_path), feat)
            except Exception as e:
                print(f"  Feature error {utt_id} ({ft}): {e}")

    print(f"  Done. Skipped={skipped} Failed={failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/raw/ASVspoof2019_synthetic_LA")
    parser.add_argument("--output_dir",  default="data/features")
    parser.add_argument("--features",    nargs="+", default=["mfcc", "lfcc", "logmel"])
    parser.add_argument("--splits",      nargs="+", default=["train", "validation", "eval"])
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    for split in args.splits:
        process_split(data_dir, output_dir, split, args.features)

    print("\nAll features extracted to:", output_dir)


if __name__ == "__main__":
    main()
