"""
Full training + evaluation pipeline runner.
Trains GMM, LightGBM (on MFCC + LFCC), and LCNN (on log-mel), then evaluates all on the eval split.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data_loader import parse_protocol, load_flat_features, ASVspoofDataset
from src.models.gmm_model import GMMClassifier
from src.models.lightgbm_model import LightGBMClassifier
from src.models.cnn_model import LCNN, CNNTrainer
from src.evaluation import (
    compute_all_metrics, print_results,
    plot_det_curve, plot_confusion_matrix, save_metrics_csv,
)

import torch
from torch.utils.data import DataLoader

DATA_DIR     = Path("data/raw/ASVspoof2019_synthetic_LA")
FEAT_DIR     = Path("data/features")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PROTO_DIR = DATA_DIR / "ASVspoof2019_LA_cm_protocols"

PROTO = {
    "train":      PROTO_DIR / "ASVspoof2019.LA.cm.train.trn.txt",
    "validation": PROTO_DIR / "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":       PROTO_DIR / "ASVspoof2019.LA.cm.eval.trl.txt",
}


def load_flat(split, feat_type):
    meta = parse_protocol(str(PROTO[split]))
    return load_flat_features(str(FEAT_DIR / split), meta, feat_type)


def load_dataset(split, feat_type):
    meta = parse_protocol(str(PROTO[split]))
    return ASVspoofDataset(str(FEAT_DIR / split), meta, feature_type=feat_type)


# ── GMM ────────────────────────────────────────────────────────────────────────
def train_eval_gmm(feat_type="mfcc"):
    print("\n" + "="*60)
    print(f"  GMM  ({feat_type.upper()})")
    print("="*60)

    X_tr, y_tr = load_flat("train", feat_type)
    X_dv, y_dv = load_flat("validation", feat_type)
    X_ev, y_ev = load_flat("eval", feat_type)

    model = GMMClassifier(n_components=64)
    model.fit(X_tr, y_tr)
    model.save(str(RESULTS_DIR / f"gmm_{feat_type}.joblib"))

    scores = model.score_samples(X_ev)
    metrics = compute_all_metrics(y_ev, scores)
    y_pred = (scores >= metrics["threshold"]).astype(int)
    print_results(f"GMM ({feat_type.upper()})", metrics, y_ev, y_pred)

    plot_confusion_matrix(y_ev, y_pred, f"GMM {feat_type.upper()}",
                          str(RESULTS_DIR / f"cm_gmm_{feat_type}.png"))
    return y_ev, scores, metrics


# ── LightGBM ───────────────────────────────────────────────────────────────────
def train_eval_lgb(feat_type="mfcc"):
    print("\n" + "="*60)
    print(f"  LightGBM  ({feat_type.upper()})")
    print("="*60)

    X_tr, y_tr = load_flat("train", feat_type)
    X_dv, y_dv = load_flat("validation", feat_type)
    X_ev, y_ev = load_flat("eval", feat_type)

    model = LightGBMClassifier()
    model.fit(X_tr, y_tr, X_dv, y_dv)
    model.save(str(RESULTS_DIR / f"lgb_{feat_type}.joblib"))

    proba = model.predict_proba(X_ev)[:, 1]
    metrics = compute_all_metrics(y_ev, proba)
    y_pred = (proba >= metrics["threshold"]).astype(int)
    print_results(f"LightGBM ({feat_type.upper()})", metrics, y_ev, y_pred)

    plot_confusion_matrix(y_ev, y_pred, f"LightGBM {feat_type.upper()}",
                          str(RESULTS_DIR / f"cm_lgb_{feat_type}.png"))
    return y_ev, proba, metrics


# ── LCNN ───────────────────────────────────────────────────────────────────────
def train_eval_lcnn(epochs=40, batch_size=64):
    print("\n" + "="*60)
    print("  LCNN  (log-mel spectrogram)")
    print("="*60)

    train_ds = load_dataset("train",      "logmel")
    dev_ds   = load_dataset("validation", "logmel")
    eval_ds  = load_dataset("eval",       "logmel")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    sample, _ = train_ds[0]
    n_mels, t_frames = sample.shape
    model = LCNN(n_mels=n_mels, time_frames=t_frames)

    save_path = str(RESULTS_DIR / "lcnn_logmel.pt")
    trainer = CNNTrainer(model, lr=1e-4)
    trainer.fit(train_loader, dev_loader, epochs=epochs, patience=8, save_path=save_path)

    _, probs, labels = trainer.evaluate(eval_loader)
    metrics = compute_all_metrics(labels, probs)
    y_pred = (probs >= metrics["threshold"]).astype(int)
    print_results("LCNN (log-mel)", metrics, labels, y_pred)

    plot_confusion_matrix(labels, y_pred, "LCNN log-mel",
                          str(RESULTS_DIR / "cm_lcnn_logmel.png"))
    return labels, probs, metrics


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    all_results  = {}
    all_metrics  = {}

    y, s, m = train_eval_gmm("mfcc")
    all_results["GMM-MFCC"]  = (y, s);  all_metrics["GMM-MFCC"]  = m

    y, s, m = train_eval_gmm("lfcc")
    all_results["GMM-LFCC"]  = (y, s);  all_metrics["GMM-LFCC"]  = m

    y, s, m = train_eval_lgb("mfcc")
    all_results["LGB-MFCC"]  = (y, s);  all_metrics["LGB-MFCC"]  = m

    y, s, m = train_eval_lgb("lfcc")
    all_results["LGB-LFCC"]  = (y, s);  all_metrics["LGB-LFCC"]  = m

    y, s, m = train_eval_lcnn(epochs=40, batch_size=64)
    all_results["LCNN"]       = (y, s);  all_metrics["LCNN"]       = m

    plot_det_curve(all_results, str(RESULTS_DIR / "det_curve.png"))
    save_metrics_csv(all_metrics, str(RESULTS_DIR / "metrics_summary.csv"))

    print("\nAll done! Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
