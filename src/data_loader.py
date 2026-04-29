"""
ASVspoof 2019 dataset loader and protocol parser.
Handles LA (Logical Access) subset used for synthetic speech detection.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from torch.utils.data import Dataset


# ASVspoof 2019 label mapping
LABEL_MAP = {"bonafide": 1, "spoof": 0}


def parse_protocol(protocol_path: str) -> pd.DataFrame:
    """
    Parse ASVspoof 2019 protocol file into a DataFrame.

    Protocol columns: SPEAKER_ID, UTTERANCE_ID, SYSTEM_ID, KEY (bonafide/spoof)
    """
    col_names = ["speaker_id", "utterance_id", "system_id", "unused", "label"]
    df = pd.read_csv(
        protocol_path,
        sep=" ",
        header=None,
        names=col_names,
    )
    df["label"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df[["utterance_id", "label", "system_id", "speaker_id"]]


def load_split_metadata(
    data_root: str, subset: str = "LA"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/dev/eval protocol files for a given subset (LA or PA).

    Returns: (train_df, dev_df, eval_df)
    """
    protocol_dir = Path(data_root) / f"ASVspoof2019_{subset}_cm_protocols"

    protocols = {
        "train": f"ASVspoof2019.{subset}.cm.train.trn.txt",
        "dev":   f"ASVspoof2019.{subset}.cm.dev.trl.txt",
        "eval":  f"ASVspoof2019.{subset}.cm.eval.trl.txt",
    }

    dfs = {}
    for split, fname in protocols.items():
        path = protocol_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Protocol file not found: {path}")
        dfs[split] = parse_protocol(str(path))

    return dfs["train"], dfs["dev"], dfs["eval"]


def get_audio_path(data_root: str, utterance_id: str, subset: str = "LA", split: str = "train") -> str:
    """Resolve full path to a .flac audio file given its utterance ID."""
    split_map = {"train": "train", "dev": "dev", "eval": "eval"}
    audio_dir = Path(data_root) / f"ASVspoof2019_{subset}_{'cm_' if subset == 'LA' else ''}train" \
        if split == "train" else \
        Path(data_root) / f"ASVspoof2019_{subset}_cm_{split_map[split]}"
    return str(audio_dir / "flac" / f"{utterance_id}.flac")


class ASVspoofDataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2019.
    Returns pre-extracted feature arrays (MFCC or LFCC) and binary labels.
    """

    def __init__(
        self,
        features_dir: str,
        metadata: pd.DataFrame,
        feature_type: str = "mfcc",
        transform=None,
    ):
        """
        Args:
            features_dir: Directory containing .npy feature files
            metadata: DataFrame with columns [utterance_id, label]
            feature_type: 'mfcc' or 'lfcc'
            transform: Optional transform applied to features
        """
        self.features_dir = Path(features_dir)
        self.feature_type = feature_type
        self.transform = transform

        # Filter to only utterances that have extracted features
        available = set(
            p.stem for p in self.features_dir.glob(f"*_{feature_type}.npy")
        )
        self.metadata = metadata[metadata["utterance_id"].isin(available)].reset_index(drop=True)

        if len(self.metadata) == 0:
            raise RuntimeError(
                f"No {feature_type} features found in {features_dir}. "
                "Run scripts/extract_features.py first."
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        feat_path = self.features_dir / f"{row['utterance_id']}_{self.feature_type}.npy"
        features = np.load(str(feat_path))

        if self.transform:
            features = self.transform(features)

        return features, int(row["label"])

    def get_labels(self) -> np.ndarray:
        return self.metadata["label"].values


def load_flat_features(
    features_dir: str,
    metadata: pd.DataFrame,
    feature_type: str = "mfcc",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features as a flat 2D matrix for sklearn/LightGBM models.
    Each audio file is summarized as mean + std of its feature frame sequence.

    Returns: (X, y) where X.shape = (n_samples, n_features*2)
    """
    features_dir = Path(features_dir)
    X, y = [], []

    for _, row in metadata.iterrows():
        feat_path = features_dir / f"{row['utterance_id']}_{feature_type}.npy"
        if not feat_path.exists():
            continue
        feat = np.load(str(feat_path))  # shape: (n_frames, n_coeffs)
        # Summarize variable-length sequence as fixed-size vector
        X.append(np.concatenate([feat.mean(axis=0), feat.std(axis=0)]))
        y.append(int(row["label"]))

    if len(X) == 0:
        raise RuntimeError(f"No features loaded from {features_dir}")

    return np.stack(X), np.array(y)
