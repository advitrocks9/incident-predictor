import numpy as np
import matplotlib.pyplot as plt


def generate_series(n_steps=5000, n_incidents=20, seed=42):
    """Generate synthetic time series with embedded incidents."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=float)

    signal = 0.5 * np.sin(2 * np.pi * t / 100) + 0.0002 * t
    signal += rng.normal(0, 0.3, n_steps)

    incidents = _place_incidents(rng, n_steps, n_incidents)

    for start, end in incidents:
        duration = end - start

        ramp_start = max(0, start - 30)
        ramp_length = start - ramp_start
        if ramp_length > 0:
            r = np.linspace(0, 1, ramp_length)
            signal[ramp_start:start] += r * 1.8
            signal[ramp_start:start] += r * rng.normal(0, 0.8, ramp_length)

        signal[start:end] += 3.0
        signal[start:end] += rng.normal(0, 0.5, duration)

    return {
        "values": signal,
        "incidents": incidents,
        "timestamps": np.arange(n_steps),
    }


def _place_incidents(rng, n_steps, n_incidents):
    usable_start = 80
    usable_end = n_steps - 200
    segment_len = (usable_end - usable_start) // n_incidents
    incidents = []

    for i in range(n_incidents):
        seg_start = usable_start + i * segment_len
        margin = max(segment_len - 50, 10)
        offset = int(rng.integers(0, margin))
        start = seg_start + offset

        if incidents and start < incidents[-1][1] + 30:
            start = incidents[-1][1] + 30

        if start >= usable_end:
            break

        duration = int(rng.integers(10, 21))
        incidents.append((start, start + duration))

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
