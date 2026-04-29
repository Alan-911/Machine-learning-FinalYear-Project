"""
Final evaluation report — loads all saved models, runs eval split,
produces DET curve, confusion matrices, and metrics CSV.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import joblib
from src.data_loader import parse_protocol, load_flat_features
from src.models.gmm_model import GMMClassifier
from src.models.lightgbm_model import LightGBMClassifier
from src.evaluation import (
    compute_all_metrics, print_results,
    plot_det_curve, plot_confusion_matrix, save_metrics_csv,
)

PROTO_DIR = Path("data/raw/ASVspoof2019_synthetic_LA/ASVspoof2019_LA_cm_protocols")
FEAT_DIR  = Path("data/features")
RESULTS   = Path("results")

def eval_meta():
    return parse_protocol(str(PROTO_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"))

def run():
    meta = eval_meta()
    all_results = {}
    all_metrics = {}

    for name, feat, model_cls, model_path in [
        ("GMM-MFCC",  "mfcc", GMMClassifier,      "gmm_mfcc.joblib"),
        ("GMM-LFCC",  "lfcc", GMMClassifier,      "gmm_lfcc.joblib"),
        ("LGB-MFCC",  "mfcc", LightGBMClassifier, "lgb_mfcc.joblib"),
        ("LGB-LFCC",  "lfcc", LightGBMClassifier, "lgb_lfcc.joblib"),
    ]:
        p = RESULTS / model_path
        if not p.exists():
            print(f"Skipping {name} — model not found")
            continue
        X, y = load_flat_features(str(FEAT_DIR / "eval"), meta, feat)
        model = model_cls.load(str(p))
        scores = (model.score_samples(X) if hasattr(model, "score_samples")
                  else model.predict_proba(X)[:, 1])
        m = compute_all_metrics(y, scores)
        y_pred = (scores >= m["threshold"]).astype(int)
        print_results(name, m, y, y_pred)
        plot_confusion_matrix(y, y_pred, name, str(RESULTS / f"cm_{name.lower().replace('-','_')}.png"))
        all_results[name] = (y, scores)
        all_metrics[name] = m

    # LCNN (scores already stored)
    lcnn_path = RESULTS / "lcnn_eval.joblib"
    if lcnn_path.exists():
        data   = joblib.load(str(lcnn_path))
        labels = data["labels"]
        probs  = data["probs"]
        m      = data["metrics"]
        y_pred = (probs >= m["threshold"]).astype(int)
        print_results("LCNN", m, labels, y_pred)
        plot_confusion_matrix(labels, y_pred, "LCNN", str(RESULTS / "cm_lcnn.png"))
        all_results["LCNN"] = (labels, probs)
        all_metrics["LCNN"] = m

    if all_results:
        plot_det_curve(all_results, str(RESULTS / "det_curve_all.png"))
        save_metrics_csv(all_metrics, str(RESULTS / "metrics_summary.csv"))

if __name__ == "__main__":
    run()
