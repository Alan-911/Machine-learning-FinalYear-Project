"""
Batch feature extraction pipeline for ASVspoof 2019.
Loads audio from HuggingFace dataset and saves MFCC, LFCC, and log-mel features.

Usage:
    python scripts/extract_features.py --output_dir data/features --split train
    python scripts/extract_features.py --output_dir data/features --split all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from src.feature_extraction import (
    extract_mfcc,
    extract_lfcc,
    extract_log_mel_spectrogram,
    SAMPLE_RATE,
)


LABEL_MAP = {"bonafide": 1, "spoof": 0}


def extract_and_save(dataset_split, output_dir: Path, feature_types: list, split_name: str):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting features for '{split_name}' split ({len(dataset_split)} samples)...")

    extractors = {
        "mfcc":   extract_mfcc,
        "lfcc":   extract_lfcc,
        "logmel": extract_log_mel_spectrogram,
    }

    skipped = 0
    for sample in tqdm(dataset_split, desc=split_name):
        utt_id = sample["utterance_id"] if "utterance_id" in sample else sample.get("id", "unknown")
        audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]

        # Resample if needed
        if sr != SAMPLE_RATE:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)

        for feat_type in feature_types:
            out_path = split_dir / f"{utt_id}_{feat_type}.npy"
            if out_path.exists():
                skipped += 1
                continue
            try:
                feat = extractors[feat_type](audio_array)
                np.save(str(out_path), feat)
            except Exception as e:
                print(f"\nWarning: failed on {utt_id} ({feat_type}): {e}")

    print(f"  Done. Skipped {skipped} already-extracted files.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/features")
    parser.add_argument("--subset", default="LA", choices=["LA", "PA"])
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "validation", "eval", "all"],
        help="Which dataset split to process",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["mfcc", "lfcc", "logmel"],
        help="Feature types to extract",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Loading ASVspoof2019 {args.subset} from HuggingFace...")
    dataset = load_dataset("asvspoof/asvspoof2019", args.subset, trust_remote_code=True)

    splits_to_process = list(dataset.keys()) if args.split == "all" else [args.split]

    for split_name in splits_to_process:
        if split_name not in dataset:
            print(f"Warning: split '{split_name}' not found. Available: {list(dataset.keys())}")
            continue
        extract_and_save(dataset[split_name], output_dir, args.features, split_name)

    print(f"\nAll features saved to {output_dir}/")


if __name__ == "__main__":
    main()
