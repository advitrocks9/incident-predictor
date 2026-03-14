import numpy as np
import matplotlib.pyplot as plt


def generate_series(n_steps=5000, n_incidents=20, seed=42):
    """Generate synthetic time series with embedded incidents."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=float)

    signal = 0.5 * np.sin(2 * np.pi * t / 100) + 0.001 * t
    signal += rng.normal(0, 0.3, n_steps)

    incidents = _place_incidents(rng, n_steps, n_incidents)

    for start, end in incidents:
        duration = end - start

        ramp_start = max(0, start - 30)
        ramp_length = start - ramp_start
        if ramp_length > 0:
            r = np.linspace(0, 1, ramp_length)
            signal[ramp_start:start] += r * 1.5
            signal[ramp_start:start] += r * rng.normal(0, 0.3, ramp_length)

        signal[start:end] += 3.0
        signal[start:end] += rng.normal(0, 0.5, duration)

    return {
        "values": signal,
        "incidents": incidents,
        "timestamps": np.arange(n_steps),
    }


def _place_incidents(rng, n_steps, n_incidents):
    min_gap = 120
    max_start = n_steps - 200
    incidents = []

    candidate = rng.integers(50, 150)
    while len(incidents) < n_incidents and candidate < max_start:
        duration = int(rng.integers(10, 21))
        incidents.append((candidate, candidate + duration))
        candidate += duration + min_gap + int(rng.integers(0, 60))

    return incidents


def plot_series(data, path=None):
    """Plot full time series with incidents shaded."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(data["timestamps"], data["values"], linewidth=0.5)

    for start, end in data["incidents"]:
        ax.axvspan(start, end, color="red", alpha=0.3)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.set_title("Synthetic Time Series with Incidents")

    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
