"""
Evaluation metrics for anti-spoofing systems.
Implements EER, Accuracy, F1, confusion matrix, and result reporting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) from binary labels and continuous scores.

    EER is the point where False Acceptance Rate (FAR) equals False Rejection Rate (FRR).
    Lower is better; 0.0 is perfect.

    Args:
        y_true: Binary ground truth (1=bonafide, 0=spoof)
        scores: Continuous scores (higher = more likely bonafide)

    Returns:
        (eer, threshold) — EER as a fraction [0, 1] and the decision threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr  # False Negative Rate = False Rejection Rate

    # Find the threshold where |FPR - FNR| is minimized
    abs_diff = np.abs(fpr - fnr)
    idx = np.argmin(abs_diff)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold)


def compute_all_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute EER, Accuracy, and F1 given ground truth and soft scores.

    If threshold is None, uses EER threshold for hard predictions.
    """
    eer, eer_threshold = compute_eer(y_true, scores)
    decision_threshold = threshold if threshold is not None else eer_threshold
    y_pred = (scores >= decision_threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    return {
        "eer": eer,
        "eer_pct": eer * 100,
        "accuracy": acc,
        "f1_score": f1,
        "threshold": decision_threshold,
    }


def print_results(model_name: str, metrics: Dict[str, float], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"\n{'='*55}")
    print(f"  {model_name} Results")
    print(f"{'='*55}")
    print(f"  EER       : {metrics['eer_pct']:.2f}%")
    print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print(f"  Threshold : {metrics['threshold']:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Spoof', 'Bonafide'])}")


def plot_det_curve(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot Detection Error Tradeoff (DET) curve for multiple models.
    results: {model_name: (y_true, scores)}
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    for name, (y_true, scores) in results.items():
        fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
        fnr = 1.0 - tpr
        ax.plot(fpr * 100, fnr * 100, label=name, linewidth=2)

    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Acceptance Rate (%)")
    ax.set_ylabel("False Rejection Rate (%)")
    ax.set_title("Detection Error Tradeoff (DET) Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"DET curve saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Spoof", "Bonafide"],
        yticklabels=["Spoof", "Bonafide"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def save_metrics_csv(all_metrics: Dict[str, Dict[str, float]], save_path: str) -> None:
    """Save a summary table of all model metrics to CSV."""
    import pandas as pd
    rows = []
    for model_name, metrics in all_metrics.items():
        rows.append({
            "Model": model_name,
            "EER (%)": f"{metrics['eer_pct']:.2f}",
            "Accuracy (%)": f"{metrics['accuracy']*100:.2f}",
            "F1-Score": f"{metrics['f1_score']:.4f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"\nMetrics summary saved to {save_path}")
    print(df.to_string(index=False))
