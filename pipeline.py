import numpy as np


def create_dataset(values, incidents, window_size=50, horizon=10):
    """Build sliding windows with binary incident labels."""
    n_steps = len(values)

    is_incident = np.zeros(n_steps, dtype=bool)
    for start, end in incidents:
        is_incident[start:end] = True

    is_buffer = np.zeros(n_steps, dtype=bool)
    for _, end in incidents:
        buf_end = min(end + horizon, n_steps)
        is_buffer[end:buf_end] = True

    incident_starts = {start for start, _ in incidents}

    windows, labels, positions = [], [], []

    # BUG: should be n_steps - horizon - 1, not n_steps - 1
    for t in range(window_size - 1, n_steps - 1):
        w_start = t - window_size + 1
        w_end = t + 1

        if np.any(is_incident[w_start:w_end]):
            continue
        if np.any(is_buffer[w_start:w_end]):
            continue

        label = int(any(s for s in incident_starts if t < s <= t + horizon))

        windows.append(values[w_start:w_end])
        labels.append(label)
        positions.append(t)

    return np.array(windows), np.array(labels), np.array(positions)


def extract_features(X_raw):
    """Compute hand-crafted features from raw windows."""
    n, w = X_raw.shape
    features = np.zeros((n, 8))
    xs = np.arange(w)

    for i, row in enumerate(X_raw):
        half = w // 2
        features[i, 0] = row.mean()
        features[i, 1] = row.std()
        features[i, 2] = np.polyfit(xs, row, 1)[0]
        features[i, 3] = row.min()
        features[i, 4] = row.max()
        features[i, 5] = row[-1]
        features[i, 6] = row.max() - row.min()
        features[i, 7] = row[half:].mean() - row[:half].mean()

    return features


def temporal_split(X, y, positions, train_frac=0.7):
    """Split by time, no shuffling."""
    split_idx = int(len(X) * train_frac)
    return (
        X[:split_idx], X[split_idx:],
        y[:split_idx], y[split_idx:],
        positions[:split_idx], positions[split_idx:],
    )
