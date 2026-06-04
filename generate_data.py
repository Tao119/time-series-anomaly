"""
generate_data.py — Synthetic time series generator for anomaly detection.

Produces three datasets, each 10,000 points with ~2% injected anomalies:
  1. ECG-like  : periodic sinusoidal with occasional spikes / missing beats
  2. Industrial: temperature sensor with trend, seasonality, sudden spikes
  3. Financial : random-walk stock price with jump anomalies (>3σ moves)

Ground-truth anomaly indices are saved alongside the series.
"""
from __future__ import annotations

import os
import numpy as np

SEED = 42
N = 10_000
ANOMALY_RATE = 0.02
OUT_DIR = os.path.dirname(__file__)


# ─────────────────────────────────────── helpers ───────────────────────────────

def _inject_spikes(rng: np.random.Generator, data: np.ndarray,
                   indices: list[int], magnitude: float = 5.0) -> np.ndarray:
    out = data.copy()
    for i in indices:
        out[i] += rng.choice([-1, 1]) * magnitude * (1 + rng.random())
    return out


def _inject_level_shifts(rng: np.random.Generator, data: np.ndarray,
                         indices: list[int], duration: int = 30,
                         shift: float = 3.0) -> np.ndarray:
    out = data.copy()
    for i in indices:
        end = min(len(data), i + duration)
        out[i:end] += rng.choice([-1, 1]) * shift
    return out


def _choose_anomaly_indices(rng: np.random.Generator, n: int,
                            rate: float, exclude: set[int] | None = None
                            ) -> list[int]:
    k = int(n * rate)
    pool = list(set(range(n)) - (exclude or set()))
    return sorted(rng.choice(pool, size=k, replace=False).tolist())


# ─────────────────────────────────────── 1. ECG-like ───────────────────────────

def generate_ecg(n: int = N, seed: int = SEED):
    """
    Sinusoidal ECG-like signal.
    Anomalies: (a) amplitude spikes, (b) 'missing beat' (flat segment).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # Base signal: 70 BPM ≈ 1 beat / 143 samples
    freq = 1 / 143
    base = np.sin(2 * np.pi * freq * t)
    # Add a QRS-like sharpening with harmonics
    base += 0.3 * np.sin(4 * np.pi * freq * t)
    base += 0.05 * rng.standard_normal(n)   # measurement noise

    n_anomalies = int(n * ANOMALY_RATE)
    spike_k = n_anomalies // 2
    flat_k  = n_anomalies - spike_k

    spike_idx = _choose_anomaly_indices(rng, n, spike_k / n)
    flat_idx  = _choose_anomaly_indices(rng, n, flat_k / n,
                                        exclude=set(spike_idx))

    data = _inject_spikes(rng, base, spike_idx, magnitude=4.0)

    # Missing beats → short flat segments
    for i in flat_idx:
        end = min(n, i + 15)
        data[i:end] = rng.uniform(-0.05, 0.05, end - i)

    anomaly_indices = sorted(spike_idx + flat_idx)
    return data, np.array(anomaly_indices, dtype=np.int64)


# ─────────────────────────────────────── 2. Industrial sensor ──────────────────

def generate_industrial(n: int = N, seed: int = SEED):
    """
    Temperature sensor: slow upward drift + daily/weekly seasonality + noise.
    Anomalies: sudden spikes and level shifts.
    """
    rng = np.random.default_rng(seed + 1)
    t = np.arange(n, dtype=float)

    trend = 0.003 * t                              # slow drift
    daily    = 5.0 * np.sin(2 * np.pi * t / 24)   # 24-step day cycle
    weekly   = 2.0 * np.sin(2 * np.pi * t / 168)  # 168-step week cycle
    noise    = rng.standard_normal(n) * 0.8

    base = 20.0 + trend + daily + weekly + noise

    n_anomalies = int(n * ANOMALY_RATE)
    spike_k  = n_anomalies * 2 // 3
    shift_k  = n_anomalies - spike_k

    spike_idx = _choose_anomaly_indices(rng, n, spike_k / n)
    shift_idx = _choose_anomaly_indices(rng, n, shift_k / n,
                                        exclude=set(spike_idx))

    data = _inject_spikes(rng, base, spike_idx, magnitude=8.0)
    data = _inject_level_shifts(rng, data, shift_idx, duration=50, shift=4.0)

    # Expand level-shift anomaly indices to cover the full shifted window
    shift_expanded: list[int] = []
    for i in shift_idx:
        shift_expanded.extend(range(i, min(n, i + 50)))

    anomaly_indices = sorted(set(spike_idx) | set(shift_expanded))
    return data, np.array(anomaly_indices, dtype=np.int64)


# ─────────────────────────────────────── 3. Financial random walk ──────────────

def generate_financial(n: int = N, seed: int = SEED):
    """
    Stock-price random walk.  Anomalies = |log-return| > 3σ (jump events).
    We inject extra jumps on top of the natural walk.
    """
    rng = np.random.default_rng(seed + 2)

    mu    = 0.0001
    sigma = 0.01
    log_returns = mu + sigma * rng.standard_normal(n)

    # Inject jumps
    n_jumps = int(n * ANOMALY_RATE)
    jump_idx = _choose_anomaly_indices(rng, n, n_jumps / n)
    for j in jump_idx:
        log_returns[j] += rng.choice([-1, 1]) * sigma * rng.uniform(4, 8)

    prices = 100.0 * np.exp(np.cumsum(log_returns))

    # Ground truth: any return > 3σ of the clean distribution
    clean_std = sigma
    anomaly_mask = np.abs(log_returns) > 3 * clean_std
    anomaly_indices = np.where(anomaly_mask)[0]

    return prices, anomaly_indices.astype(np.int64)


# ─────────────────────────────────────── main ──────────────────────────────────

def generate_all(out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    generators = {
        "ecg":        generate_ecg,
        "industrial": generate_industrial,
        "financial":  generate_financial,
    }

    summary: dict[str, dict] = {}
    for name, gen_fn in generators.items():
        data, anomaly_idx = gen_fn()
        np.save(os.path.join(out_dir, f"{name}_data.npy"), data)
        np.save(os.path.join(out_dir, f"{name}_anomaly_idx.npy"), anomaly_idx)

        rate = len(anomaly_idx) / len(data) * 100
        print(f"[{name:12s}] n={len(data):,}  anomalies={len(anomaly_idx):,}"
              f"  rate={rate:.1f}%  range=[{data.min():.2f}, {data.max():.2f}]")
        summary[name] = {
            "n": len(data),
            "n_anomalies": int(len(anomaly_idx)),
            "anomaly_rate": float(rate),
        }

    return summary


if __name__ == "__main__":
    print("Generating synthetic time-series datasets …")
    generate_all()
    print("Done — .npy files saved to current directory.")
