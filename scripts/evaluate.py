"""
Model evaluation script — runs trained models on the eval split and reports results.

Usage:
    python scripts/evaluate.py \
        --features_dir data/features \
        --protocol_dir data/raw/ASVspoof2019_LA_cm_protocols \
        --results_dir results/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_loader import load_split_metadata, load_flat_features, ASVspoofDataset
from src.evaluation import (
    compute_all_metrics,
    print_results,
    plot_det_curve,
    plot_confusion_matrix,
    save_metrics_csv,
)


def evaluate_gmm(results_dir: Path, features_dir: Path, eval_meta, feature_type: str):
    from src.models.gmm_model import GMMClassifier
    model_path = results_dir / f"gmm_{feature_type}.joblib"
    if not model_path.exists():
        print(f"GMM model not found at {model_path}. Skipping.")
        return None, None, None

    model = GMMClassifier.load(str(model_path))
    X_eval, y_eval = load_flat_features(str(features_dir / "eval"), eval_meta, feature_type)
    scores = model.score_samples(X_eval)
    metrics = compute_all_metrics(y_eval, scores)
    y_pred = (scores >= metrics["threshold"]).astype(int)
    print_results("GMM", metrics, y_eval, y_pred)
    return y_eval, scores, metrics


def evaluate_lightgbm(results_dir: Path, features_dir: Path, eval_meta, feature_type: str):
    from src.models.lightgbm_model import LightGBMClassifier
    model_path = results_dir / f"lightgbm_{feature_type}.joblib"
    if not model_path.exists():
        print(f"LightGBM model not found at {model_path}. Skipping.")
        return None, None, None

    model = LightGBMClassifier.load(str(model_path))
    X_eval, y_eval = load_flat_features(str(features_dir / "eval"), eval_meta, feature_type)
    proba = model.predict_proba(X_eval)[:, 1]
    metrics = compute_all_metrics(y_eval, proba)
    y_pred = (proba >= metrics["threshold"]).astype(int)
    print_results("LightGBM", metrics, y_eval, y_pred)
    return y_eval, proba, metrics


def evaluate_cnn(results_dir: Path, features_dir: Path, eval_meta, arch: str = "lcnn"):
    from src.models.cnn_model import LCNN, ResNetDetector, CNNTrainer
    model_path = results_dir / f"cnn_{arch}_logmel.pt"
    if not model_path.exists():
        print(f"CNN model not found at {model_path}. Skipping.")
        return None, None, None

    eval_dataset = ASVspoofDataset(str(features_dir / "eval"), eval_meta, feature_type="logmel")
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4)

    sample, _ = eval_dataset[0]
    n_mels, time_frames = sample.shape
    model = LCNN(n_mels=n_mels, time_frames=time_frames) if arch == "lcnn" \
        else ResNetDetector(pretrained=False)
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))

    trainer = CNNTrainer(model)
    _, probs, labels = trainer.evaluate(eval_loader)
    metrics = compute_all_metrics(labels, probs)
    y_pred = (probs >= metrics["threshold"]).astype(int)
    print_results(f"CNN ({arch.upper()})", metrics, labels, y_pred)
    return labels, probs, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--protocol_dir", required=True)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--feature_type", default="mfcc", choices=["mfcc", "lfcc"])
    parser.add_argument("--cnn_arch", default="lcnn", choices=["lcnn", "resnet"])
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    _, _, eval_meta = load_split_metadata(
        str(Path(args.protocol_dir).parent), subset="LA"
    )
    print(f"Eval set: {len(eval_meta)} samples")

    all_results = {}
    all_metrics = {}

    y_gmm, s_gmm, m_gmm = evaluate_gmm(results_dir, features_dir, eval_meta, args.feature_type)
    if y_gmm is not None:
        all_results["GMM"] = (y_gmm, s_gmm)
        all_metrics["GMM"] = m_gmm

    y_lgb, s_lgb, m_lgb = evaluate_lightgbm(results_dir, features_dir, eval_meta, args.feature_type)
    if y_lgb is not None:
        all_results["LightGBM"] = (y_lgb, s_lgb)
        all_metrics["LightGBM"] = m_lgb

    y_cnn, s_cnn, m_cnn = evaluate_cnn(results_dir, features_dir, eval_meta, args.cnn_arch)
    if y_cnn is not None:
        all_results[f"CNN-{args.cnn_arch.upper()}"] = (y_cnn, s_cnn)
        all_metrics[f"CNN-{args.cnn_arch.upper()}"] = m_cnn

    if all_results:
        plot_det_curve(all_results, save_path=str(results_dir / "det_curve.png"))
        save_metrics_csv(all_metrics, str(results_dir / "eval_metrics_summary.csv"))

        for name, (y_true, scores) in all_results.items():
            threshold = all_metrics[name]["threshold"]
            y_pred = (scores >= threshold).astype(int)
            plot_confusion_matrix(
                y_true, y_pred, name,
                save_path=str(results_dir / f"confusion_{name.lower().replace(' ', '_')}.png")
            )


if __name__ == "__main__":
    main()
