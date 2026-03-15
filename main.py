import argparse
import os

from data import generate_series
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

    feat_train = extract_features(X_train)
    feat_test = extract_features(X_test)

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
    gb.fit(X_train, y_train)
    models["Gradient Boosting"] = (
        gb.predict(X_test),
        gb.predict_proba(X_test),
    )

    # TODO: evaluation and plotting


if __name__ == "__main__":
    main()
