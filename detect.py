"""
detect.py — Unified anomaly detection pipeline.

Public API
----------
    result = detect_anomalies(series, method="autoencoder")
    # result = {"anomaly_indices": list[int], "scores": np.ndarray, "threshold": float}

Supported methods: "autoencoder", "isolation_forest", "prophet"

Saves a PNG plot with anomalies highlighted for each call.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(__file__)

# ─────────────────────────────────────── main API ──────────────────────────────

def detect_anomalies(
    series: np.ndarray,
    method: str = "autoencoder",
    dataset_name: str = "series",
    save_plot: bool = True,
    # autoencoder kwargs
    ae_epochs: int = 15,
    ae_window: int = 50,
    # isolation forest kwargs
    if_window: int = 50,
    if_step: int = 5,
    if_contamination: float = 0.02,
    # prophet kwargs
    prophet_trend_window: int = 24,
    prophet_k_iqr: float = 3.0,
) -> dict:
    """
    Detect anomalies in `series` using the specified method.

    Parameters
    ----------
    series       : 1-D numpy array, the time series
    method       : one of "autoencoder", "isolation_forest", "prophet"
    dataset_name : used for plot filename
    save_plot    : if True, saves a PNG with anomalies highlighted

    Returns
    -------
    dict with keys:
        "anomaly_indices" : list[int]
        "scores"          : np.ndarray  (per timestep, higher = more anomalous)
        "threshold"       : float
    """
    series = np.asarray(series, dtype=float)
    method = method.lower().strip()

    if method == "autoencoder":
        result = _run_autoencoder(series, ae_epochs, ae_window)
    elif method in ("isolation_forest", "iforest", "if"):
        result = _run_isolation_forest(series, if_window, if_step, if_contamination)
    elif method in ("prophet", "prophet_simple"):
        result = _run_prophet(series, prophet_trend_window, prophet_k_iqr)
    else:
        raise ValueError(f"Unknown method '{method}'. "
                         "Choose from: autoencoder, isolation_forest, prophet")

    if save_plot:
        _plot_result(series, result, method, dataset_name)

    return result


# ─────────────────────────────────────── per-method runners ────────────────────

def _run_autoencoder(series: np.ndarray, epochs: int, window: int) -> dict:
    import sys
    sys.path.insert(0, OUT_DIR)
    from models.autoencoder import AnomalyAutoencoder

    # Split: first 70% for training (assumed mostly normal)
    split = int(len(series) * 0.7)
    train = series[:split]

    ae = AnomalyAutoencoder(window=window, epochs=epochs, batch_size=256)
    ae.fit(train, verbose=False)
    return ae.predict(series)


def _run_isolation_forest(series: np.ndarray, window: int, step: int,
                          contamination: float) -> dict:
    import sys
    sys.path.insert(0, OUT_DIR)
    from models.isolation_forest import IsolationForestDetector

    split = int(len(series) * 0.7)
    detector = IsolationForestDetector(
        window=window, step=step, contamination=contamination
    )
    detector.fit(series[:split])
    return detector.predict(series)


def _run_prophet(series: np.ndarray, trend_window: int, k_iqr: float) -> dict:
    import sys
    sys.path.insert(0, OUT_DIR)
    from models.prophet_simple import ProphetSimple

    split = int(len(series) * 0.7)
    model = ProphetSimple(trend_window=trend_window, k_iqr=k_iqr, max_periods=3)
    model.fit(series[:split])
    return model.predict(series)


# ─────────────────────────────────────── visualization ─────────────────────────

def _plot_result(series: np.ndarray, result: dict, method: str,
                 dataset_name: str) -> None:
    anomaly_indices = result["anomaly_indices"]
    scores = result["scores"]
    threshold = result["threshold"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # --- top: raw series with anomalies
    ax = axes[0]
    ax.plot(series, color="steelblue", linewidth=0.6, label="Time series", alpha=0.9)
    if anomaly_indices:
        ax.scatter(anomaly_indices, series[anomaly_indices],
                   color="crimson", s=15, zorder=5, label=f"Anomalies ({len(anomaly_indices)})")
    ax.set_ylabel("Value")
    ax.set_title(f"{dataset_name}  |  method = {method}  |  anomalies = {len(anomaly_indices)}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # --- bottom: anomaly scores
    ax2 = axes[1]
    ax2.plot(scores, color="darkorange", linewidth=0.6, label="Anomaly score")
    ax2.axhline(threshold, color="crimson", linestyle="--", linewidth=1.0,
                label=f"Threshold = {threshold:.4f}")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Score")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"detect_{dataset_name}_{method}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


# ─────────────────────────────────────── CLI / demo ────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, OUT_DIR)
    from generate_data import generate_ecg, generate_industrial, generate_financial

    datasets = {
        "ecg":        generate_ecg,
        "industrial": generate_industrial,
        "financial":  generate_financial,
    }
    methods = ["autoencoder", "isolation_forest", "prophet"]

    for ds_name, gen_fn in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        data, gt_idx = gen_fn()
        np.save(os.path.join(OUT_DIR, f"{ds_name}_data.npy"), data)
        np.save(os.path.join(OUT_DIR, f"{ds_name}_anomaly_idx.npy"), gt_idx)

        for method in methods:
            print(f"  Running {method} …")
            result = detect_anomalies(
                data,
                method=method,
                dataset_name=ds_name,
                save_plot=True,
                ae_epochs=10,
            )
            print(f"    Detected: {len(result['anomaly_indices'])}  "
                  f"GT: {len(gt_idx)}")

    print("\nDone.")
