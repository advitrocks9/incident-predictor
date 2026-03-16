import argparse
import os

import numpy as np

from data import generate_series, plot_series
from evaluate import plot_confusion_matrix, plot_pr_curves, print_metrics, threshold_analysis
from models import GradientBoostingModel, LogisticRegressionBaseline, StaticThresholdBaseline
from pipeline import create_dataset, extract_features, temporal_split


def parse_args():
    parser = argparse.ArgumentParser(description="Incident prediction from time-series metrics")
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Config: W={args.window_size}, H={args.horizon}, seed={args.seed}")

    data = generate_series(n_steps=5000, n_incidents=20, seed=args.seed)
    print(f"Generated {len(data['values'])} steps with {len(data['incidents'])} incidents")

    X, y, positions = create_dataset(
        data["values"], data["incidents"],
        window_size=args.window_size, horizon=args.horizon,
    )
    print(f"Windows: {len(y)} total, {y.sum()} positive ({y.mean():.1%})")

    X_train, X_test, y_train, y_test, _, _ = temporal_split(X, y, positions)
    print(f"Train: {len(y_train)} ({y_train.sum()} pos) | Test: {len(y_test)} ({y_test.sum()} pos)")

    plot_series(data, path=os.path.join(args.output_dir, "series.png"))

    feat_train = extract_features(X_train)
    feat_test = extract_features(X_test)

    gb_train = np.hstack([X_train, feat_train])
    gb_test = np.hstack([X_test, feat_test])

    models = {}

    threshold_bl = StaticThresholdBaseline()
    threshold_bl.fit(X_train, y_train)
    models["Static Threshold"] = (
        threshold_bl.predict(X_test),
        threshold_bl.predict_proba(X_test),
    )

    lr = LogisticRegressionBaseline()
    lr.fit(feat_train, y_train)
    models["Logistic Regression"] = (
        lr.predict(feat_test),
        lr.predict_proba(feat_test),
    )

    gb = GradientBoostingModel()
    gb.fit(gb_train, y_train)
    models["Gradient Boosting"] = (
        gb.predict(gb_test),
        gb.predict_proba(gb_test),
    )

    print("\n" + "=" * 60)
    pr_results = {}
    for name, (preds, probs) in models.items():
        print_metrics(y_test, preds, probs, name)
        pr_results[name] = (y_test, probs)
        print()

    plot_pr_curves(pr_results, os.path.join(args.output_dir, "pr_curve.png"))
    plot_confusion_matrix(
        y_test, models["Gradient Boosting"][0], "Gradient Boosting",
        os.path.join(args.output_dir, "confusion_matrix.png"),
    )
    threshold_analysis(
        y_test, models["Gradient Boosting"][1], "Gradient Boosting",
        path=os.path.join(args.output_dir, "threshold_analysis.png"),
    )

    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
