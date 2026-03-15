import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)


def print_metrics(y_true, y_pred, y_prob, name):
    """Print classification metrics for a model."""
    print(f"[{name}]")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  F1:        {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  AP:        {average_precision_score(y_true, y_prob):.3f}")
    print(f"  Predicted pos: {y_pred.sum()} / Total: {len(y_true)}")


def plot_pr_curves(results, path):
    """Plot precision-recall curves for all models."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (y_true, y_prob) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    ax.grid(True)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, name, path):
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
