"""
Unified training entrypoint for all three models.

Usage:
    python scripts/train.py --model all --features_dir data/features
    python scripts/train.py --model lightgbm --feature_type mfcc
    python scripts/train.py --model cnn --feature_type logmel --epochs 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_loader import load_flat_features, ASVspoofDataset
from src.models.gmm_model import GMMClassifier
from src.models.lightgbm_model import LightGBMClassifier
from src.models.cnn_model import LCNN, ResNetDetector, CNNTrainer
from src.evaluation import compute_all_metrics, print_results


RESULTS_DIR = Path("results")


def load_metadata_from_features(features_dir: Path, split: str, feature_type: str):
    """Build a metadata DataFrame from available .npy files in a split directory."""
    import pandas as pd

    split_dir = features_dir / split
    rows = []
    for p in split_dir.glob(f"*_{feature_type}.npy"):
        utt_id = p.stem.replace(f"_{feature_type}", "")
        # Label is encoded in filename: LA_T_ prefix = train bonafide, LA_E_ = eval
        # Actual label requires protocol file; here we try to infer from naming convention
        # For proper use: load protocol files via src.data_loader.parse_protocol
        rows.append({"utterance_id": utt_id, "label": -1})  # label TBD from protocol
    return pd.DataFrame(rows)


def train_gmm(features_dir: Path, feature_type: str, train_meta, dev_meta):
    print("\n" + "="*50)
    print("Training GMM Classifier")
    print("="*50)

    X_train, y_train = load_flat_features(str(features_dir / "train"), train_meta, feature_type)
    X_dev, y_dev = load_flat_features(str(features_dir / "validation"), dev_meta, feature_type)

    model = GMMClassifier(n_components=128)
    model.fit(X_train, y_train)

    scores = model.score_samples(X_dev)
    metrics = compute_all_metrics(y_dev, scores)
    y_pred = (scores >= metrics["threshold"]).astype(int)
    print_results("GMM", metrics, y_dev, y_pred)

    RESULTS_DIR.mkdir(exist_ok=True)
    model.save(str(RESULTS_DIR / f"gmm_{feature_type}.joblib"))
    return metrics


def train_lightgbm(features_dir: Path, feature_type: str, train_meta, dev_meta):
    print("\n" + "="*50)
    print("Training LightGBM Classifier")
    print("="*50)

    X_train, y_train = load_flat_features(str(features_dir / "train"), train_meta, feature_type)
    X_dev, y_dev = load_flat_features(str(features_dir / "validation"), dev_meta, feature_type)

    model = LightGBMClassifier()
    model.fit(X_train, y_train, X_dev, y_dev)

    proba = model.predict_proba(X_dev)[:, 1]
    metrics = compute_all_metrics(y_dev, proba)
    y_pred = (proba >= metrics["threshold"]).astype(int)
    print_results("LightGBM", metrics, y_dev, y_pred)

    RESULTS_DIR.mkdir(exist_ok=True)
    model.save(str(RESULTS_DIR / f"lightgbm_{feature_type}.joblib"))
    return metrics


def train_cnn(
    features_dir: Path,
    feature_type: str,
    train_meta,
    dev_meta,
    arch: str = "lcnn",
    epochs: int = 50,
    batch_size: int = 64,
):
    print("\n" + "="*50)
    print(f"Training CNN ({arch.upper()})")
    print("="*50)

    train_dataset = ASVspoofDataset(
        str(features_dir / "train"), train_meta, feature_type=feature_type
    )
    dev_dataset = ASVspoofDataset(
        str(features_dir / "validation"), dev_meta, feature_type=feature_type
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    sample, _ = train_dataset[0]
    n_mels, time_frames = sample.shape

    model = LCNN(n_mels=n_mels, time_frames=time_frames) if arch == "lcnn" \
        else ResNetDetector(pretrained=False)

    save_path = str(RESULTS_DIR / f"cnn_{arch}_{feature_type}.pt")
    RESULTS_DIR.mkdir(exist_ok=True)

    trainer = CNNTrainer(model, lr=1e-4)
    trainer.fit(train_loader, dev_loader, epochs=epochs, patience=10, save_path=save_path)

    _, probs, labels = trainer.evaluate(dev_loader)
    metrics = compute_all_metrics(labels, probs)
    y_pred = (probs >= metrics["threshold"]).astype(int)
    print_results(f"CNN ({arch.upper()})", metrics, labels, y_pred)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["gmm", "lightgbm", "cnn", "all"])
    parser.add_argument("--feature_type", default="mfcc", choices=["mfcc", "lfcc", "logmel"])
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--cnn_arch", default="lcnn", choices=["lcnn", "resnet"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--protocol_dir",
        default=None,
        help="Path to ASVspoof2019 protocol files. Required to load labels.",
    )
    args = parser.parse_args()

    features_dir = Path(args.features_dir)

    # Load metadata from protocol files
    if args.protocol_dir is None:
        print(
            "ERROR: --protocol_dir is required.\n"
            "Provide the path to the ASVspoof2019_LA_cm_protocols directory.\n"
            "Example: python scripts/train.py --protocol_dir data/raw/ASVspoof2019_LA_cm_protocols"
        )
        sys.exit(1)

    from src.data_loader import load_split_metadata
    train_meta, dev_meta, eval_meta = load_split_metadata(
        str(Path(args.protocol_dir).parent), subset="LA"
    )

    print(f"Train: {len(train_meta)} | Dev: {len(dev_meta)} | Eval: {len(eval_meta)}")

    all_metrics = {}

    if args.model in ("gmm", "all"):
        ft = args.feature_type if args.feature_type != "logmel" else "mfcc"
        all_metrics["GMM"] = train_gmm(features_dir, ft, train_meta, dev_meta)

    if args.model in ("lightgbm", "all"):
        ft = args.feature_type if args.feature_type != "logmel" else "mfcc"
        all_metrics["LightGBM"] = train_lightgbm(features_dir, ft, train_meta, dev_meta)

    if args.model in ("cnn", "all"):
        all_metrics[f"CNN-{args.cnn_arch.upper()}"] = train_cnn(
            features_dir,
            feature_type="logmel",
            train_meta=train_meta,
            dev_meta=dev_meta,
            arch=args.cnn_arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    # Summary
    if all_metrics:
        from src.evaluation import save_metrics_csv
        save_metrics_csv(all_metrics, str(RESULTS_DIR / "metrics_summary.csv"))


if __name__ == "__main__":
    main()
