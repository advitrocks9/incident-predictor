import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


class StaticThresholdBaseline:
    """Alert if last window value exceeds a learned threshold."""

    def fit(self, X_raw, y):
        last_vals = X_raw[:, -1]
        self.min_val_ = last_vals.min()
        self.max_val_ = last_vals.max()

        candidates = np.percentile(last_vals, np.linspace(70, 99, 50))
        best_f1, best_thresh = 0, candidates[0]

        for thresh in candidates:
            preds = (last_vals > thresh).astype(int)
            tp = ((preds == 1) & (y == 1)).sum()
            fp = ((preds == 1) & (y == 0)).sum()
            fn = ((preds == 0) & (y == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        self.threshold_ = best_thresh
        print(f"  Static threshold: {self.threshold_:.3f} (F1={best_f1:.3f})")

    def predict(self, X_raw):
        return (X_raw[:, -1] > self.threshold_).astype(int)

    def predict_proba(self, X_raw):
        scaled = (X_raw[:, -1] - self.min_val_) / (self.max_val_ - self.min_val_ + 1e-8)
        return np.clip(scaled, 0, 1)


class LogisticRegressionBaseline:

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000)

    def fit(self, X_features, y):
        X_scaled = self.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y)

    def predict(self, X_features):
        return self.model.predict(self.scaler.transform(X_features))

    def predict_proba(self, X_features):
        return self.model.predict_proba(self.scaler.transform(X_features))[:, 1]


class GradientBoostingModel:

    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1, random_state=42,
        )

    def fit(self, X_raw, y):
        weights = compute_sample_weight("balanced", y)
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        print(f"  Class distribution: {n_neg} neg / {n_pos} pos")
        print(f"  Weight ratio: {weights[y == 1].mean():.1f}x")
        self.model.fit(X_raw, y, sample_weight=weights)

    def predict(self, X_raw):
        return self.model.predict(X_raw)

    def predict_proba(self, X_raw):
        return self.model.predict_proba(X_raw)[:, 1]
