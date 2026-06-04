"""
benchmark.py — Comprehensive Anomaly Detection Benchmark

Compares 5 methods on 3 synthetic datasets:
  Methods : Prophet, IsolationForest, LSTM_AE, TCN, LSTM_VAE
  Datasets: ECG, Industrial, Financial (each 10,000 points, ~2% anomaly rate)
  Metrics : Precision, Recall, F1, ROC-AUC

Outputs:
  benchmark_results.json  — full metrics table
  benchmark_table.png     — formatted table image
"""
from __future__ import annotations

import sys
import os
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────── sklearn-like AUC ──────────────────────

def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn."""
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.5
    pos_scores = scores[pos_idx]
    neg_scores = scores[neg_idx]
    # Mann-Whitney U statistic
    count = 0.0
    for ps in pos_scores:
        count += float(np.sum(ps > neg_scores))
        count += 0.5 * float(np.sum(ps == neg_scores))
    return count / (len(pos_scores) * len(neg_scores))


def compute_metrics(anomaly_indices: list[int], scores: np.ndarray,
                    gt_indices: np.ndarray, n: int) -> dict[str, float]:
    """Compute Precision, Recall, F1, ROC-AUC."""
    gt_set = set(gt_indices.tolist())
    pred_set = set(anomaly_indices)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)

    # Binary labels for AUC
    labels = np.zeros(n, dtype=np.int32)
    labels[list(gt_set)] = 1

    # Normalise scores to [0,1]
    s_min, s_max = scores.min(), scores.max()
    norm_scores = (scores - s_min) / (s_max - s_min + 1e-12)
    auc = roc_auc(norm_scores, labels)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "auc":       round(auc, 4),
    }


# ─────────────────────────────────────── dataset loading ───────────────────────

def load_datasets(base_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from generate_data import generate_ecg, generate_industrial, generate_financial
    names = {"ECG": generate_ecg, "Industrial": generate_industrial,
             "Financial": generate_financial}
    datasets = {}
    for name, fn in names.items():
        data, gt_idx = fn()
        datasets[name] = (data, gt_idx)
        print(f"  [{name:12s}] n={len(data):,}  anomalies={len(gt_idx):,}"
              f"  rate={len(gt_idx)/len(data)*100:.1f}%")
    return datasets


# ─────────────────────────────────────── method runners ────────────────────────

def run_prophet(train: np.ndarray, test: np.ndarray,
                gt_idx: np.ndarray) -> dict[str, float]:
    from models.prophet_simple import ProphetSimple
    model = ProphetSimple(trend_window=24, k_iqr=3.0, max_periods=3)
    model.fit(train)
    result = model.predict(test)
    return compute_metrics(result["anomaly_indices"], result["scores"],
                            gt_idx, len(test))


def run_isolation_forest(train: np.ndarray, test: np.ndarray,
                          gt_idx: np.ndarray) -> dict[str, float]:
    from models.isolation_forest import IsolationForestDetector
    detector = IsolationForestDetector(window=50, step=10, contamination=0.02)
    detector.fit(train)
    result = detector.predict(test)
    return compute_metrics(result["anomaly_indices"], result["scores"],
                            gt_idx, len(test))


def run_lstm_ae(train: np.ndarray, test: np.ndarray,
                gt_idx: np.ndarray, epochs: int = 10) -> dict[str, float]:
    from models.autoencoder import AnomalyAutoencoder
    ae = AnomalyAutoencoder(window=50, hidden=32, latent=16,
                            epochs=epochs, batch_size=128)
    ae.fit(train, verbose=False)
    result = ae.predict(test)
    return compute_metrics(result["anomaly_indices"], result["scores"],
                            gt_idx, len(test))


def run_tcn(train: np.ndarray, test: np.ndarray,
            gt_idx: np.ndarray, epochs: int = 10) -> dict[str, float]:
    from models.tcn_anomaly import TCNAnomalyDetector
    model = TCNAnomalyDetector(window=50, stride=25, enc_channels=32,
                                bottleneck=16, epochs=epochs, batch_size=128)
    model.fit(train, verbose=False)
    result = model.predict(test)
    return compute_metrics(result["anomaly_indices"], result["scores"],
                            gt_idx, len(test))


def run_lstm_vae(train: np.ndarray, test: np.ndarray,
                 gt_idx: np.ndarray, epochs: int = 10) -> dict[str, float]:
    from models.lstm_vae import LSTMVAEDetector
    model = LSTMVAEDetector(window=50, stride=25, hidden_size=64,
                             latent_dim=32, beta=0.1, epochs=epochs,
                             batch_size=256)
    model.fit(train, verbose=False)
    result = model.predict(test)
    return compute_metrics(result["anomaly_indices"], result["scores"],
                            gt_idx, len(test))


# ─────────────────────────────────────── table printing ────────────────────────

def print_table(dataset_name: str, n: int, anomaly_rate: float,
                results: dict[str, dict[str, float]]) -> None:
    print(f"\nDataset: {dataset_name} ({n:,} points, {anomaly_rate:.1f}% anomaly rate)")
    header = f"{'Method':<20}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'AUC':>7}"
    print(header)
    print("-" * len(header))
    for method, m in results.items():
        print(f"  {method:<18}  {m['precision']:>9.4f}  {m['recall']:>7.4f}"
              f"  {m['f1']:>7.4f}  {m['auc']:>7.4f}")


def save_table_png(all_results: dict, save_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        methods = ["Prophet", "IsolationForest", "LSTM_AE", "TCN", "LSTM_VAE"]
        datasets = list(all_results.keys())
        metrics_names = ["Precision", "Recall", "F1", "AUC"]
        metric_keys   = ["precision", "recall", "f1", "auc"]

        n_ds = len(datasets)
        n_met = len(metrics_names)
        n_methods = len(methods)

        fig, axes = plt.subplots(n_ds, n_met,
                                  figsize=(4 * n_met, 3.5 * n_ds),
                                  squeeze=False)
        fig.suptitle("Anomaly Detection Benchmark", fontsize=16, fontweight="bold")

        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

        for di, ds_name in enumerate(datasets):
            ds_res = all_results[ds_name]
            for mi, (m_name, m_key) in enumerate(zip(metrics_names, metric_keys)):
                ax = axes[di][mi]
                vals = [ds_res.get(meth, {}).get(m_key, 0.0) for meth in methods]
                bars = ax.bar(methods, vals, color=colors, edgecolor="white", linewidth=0.8)
                ax.set_title(f"{ds_name} — {m_name}", fontsize=9)
                ax.set_ylim(0, 1.05)
                ax.set_xticks(range(len(methods)))
                ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=7)
                ax.yaxis.set_tick_params(labelsize=8)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            val + 0.02, f"{val:.3f}",
                            ha="center", va="bottom", fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved table plot: {save_path}")
    except Exception as exc:
        print(f"  Could not save PNG (matplotlib error): {exc}")


# ─────────────────────────────────────── main ──────────────────────────────────

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    EPOCHS = 10   # quick test; increase to 30+ for better performance

    print("Loading datasets …")
    datasets = load_datasets(BASE_DIR)

    METHOD_RUNNERS = {
        "Prophet":         run_prophet,
        "IsolationForest": run_isolation_forest,
        "LSTM_AE":         lambda tr, te, gt: run_lstm_ae(tr, te, gt, EPOCHS),
        "TCN":             lambda tr, te, gt: run_tcn(tr, te, gt, EPOCHS),
        "LSTM_VAE":        lambda tr, te, gt: run_lstm_vae(tr, te, gt, EPOCHS),
    }

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for ds_name, (data, gt_idx) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print("=" * 60)

        # Use first 70% as train (assumed mostly normal), full series as test
        split = int(len(data) * 0.70)
        train_data = data[:split]
        test_data  = data

        ds_results: dict[str, dict[str, float]] = {}
        for method_name, runner in METHOD_RUNNERS.items():
            print(f"  Running {method_name} …", end=" ", flush=True)
            t0 = time.time()
            try:
                metrics = runner(train_data, test_data, gt_idx)
                elapsed = time.time() - t0
                metrics["time_s"] = round(elapsed, 2)
                print(f"done ({elapsed:.1f}s)"
                      f"  P={metrics['precision']:.3f}"
                      f"  R={metrics['recall']:.3f}"
                      f"  F1={metrics['f1']:.3f}"
                      f"  AUC={metrics['auc']:.3f}")
            except Exception as exc:
                print(f"ERROR: {exc}")
                metrics = {"precision": 0.0, "recall": 0.0,
                           "f1": 0.0, "auc": 0.5, "time_s": 0.0}
            ds_results[method_name] = metrics

        all_results[ds_name] = ds_results
        anomaly_rate = len(gt_idx) / len(data) * 100
        print_table(ds_name, len(data), anomaly_rate, ds_results)

    # ── Save JSON ──
    json_path = os.path.join(BASE_DIR, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_path}")

    # ── Save PNG ──
    png_path = os.path.join(BASE_DIR, "benchmark_table.png")
    print("Generating table plot …")
    save_table_png(all_results, png_path)

    # ── Final summary ──
    print("\n" + "=" * 65)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 65)
    for ds_name, ds_results in all_results.items():
        n, gt_idx = len(datasets[ds_name][0]), datasets[ds_name][1]
        anomaly_rate = len(gt_idx) / n * 100
        print(f"\nDataset: {ds_name} ({n:,} points, {anomaly_rate:.1f}% anomaly rate)")
        print(f"  {'Method':<20}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'AUC':>7}")
        print(f"  {'-'*56}")
        for method, m in ds_results.items():
            print(f"  {method:<20}  {m['precision']:>9.4f}  "
                  f"{m['recall']:>7.4f}  {m['f1']:>7.4f}  {m['auc']:>7.4f}")
    print("=" * 65)
