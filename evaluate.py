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


def threshold_analysis(y_true, y_prob, name, path=None):
    """Evaluate metrics across decision thresholds."""
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    print(f"Threshold Analysis - [{name}]")
    print(f"  {'Thresh':<8}{'Prec':<8}{'Rec':<8}{'F1':<8}{'FPR':<8}{'Alerts'}")

    rows = []
    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        alerts = preds.sum()

        print(f"  {thresh:<8.2f}{prec:<8.3f}{rec:<8.3f}{f1:<8.3f}{fpr:<8.3f}{alerts}")
        rows.append({"threshold": thresh, "precision": prec, "recall": rec,
                      "f1": f1, "fpr": fpr, "alerts": alerts})

    if path:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, [r["precision"] for r in rows], "o-", label="Precision")
        ax.plot(thresholds, [r["recall"] for r in rows], "s-", label="Recall")
        ax.plot(thresholds, [r["f1"] for r in rows], "^-", label="F1")

        best_f1_idx = max(range(len(rows)), key=lambda i: rows[i]["f1"])
        ax.axvline(thresholds[best_f1_idx], color="gray", linestyle="--", alpha=0.7,
                    label=f"Best F1 @ {thresholds[best_f1_idx]:.1f}")

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"Threshold Analysis - {name}")
        ax.legend()
        ax.grid(True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return rows
