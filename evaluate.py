"""
evaluate.py — Evaluation metrics and full benchmark for anomaly detection.

Metrics
-------
- Precision, Recall, F1
- ROC-AUC
- Point-Adjust F1: an anomaly segment is "correct" if any point within
  the ground-truth range is detected.

Runs all 3 methods on all 3 datasets and saves results_summary.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from generate_data import generate_ecg, generate_industrial, generate_financial
from detect import detect_anomalies

OUT_DIR = os.path.dirname(__file__)


# ─────────────────────────────────────── metric helpers ────────────────────────

def _binary_label_array(anomaly_indices: list[int] | np.ndarray,
                        n: int) -> np.ndarray:
    labels = np.zeros(n, dtype=bool)
    if len(anomaly_indices):
        labels[np.asarray(anomaly_indices, dtype=int)] = True
    return labels


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Trapezoidal ROC-AUC (pure NumPy)."""
    y_true = y_true.astype(bool)
    n_pos = y_true.sum()
    n_neg = (~y_true).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5

    thresholds = np.unique(scores)[::-1]
    tpr_list, fpr_list = [0.0], [0.0]
    for thr in thresholds:
        pred = scores >= thr
        tp = float((y_true & pred).sum())
        fp = float((~y_true & pred).sum())
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    return float(np.trapz(tpr_list, fpr_list))


def point_adjust_f1(y_true: np.ndarray, y_pred: np.ndarray,
                    window: int = 10) -> float:
    """
    Point-adjust F1: for each ground-truth anomaly run, if any predicted
    point falls within [start - window, end + window], all GT points
    in the run are counted as detected.
    """
    n = len(y_true)
    # Find contiguous anomaly segments in ground truth
    segments = _get_segments(y_true)

    y_pred_adj = np.zeros(n, dtype=bool)
    for seg_start, seg_end in segments:
        lo = max(0, seg_start - window)
        hi = min(n, seg_end + window)
        if y_pred[lo:hi].any():
            y_pred_adj[seg_start:seg_end] = True

    return precision_recall_f1(y_true, y_pred_adj)["f1"]


def _get_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Returns list of (start, end+1) for True runs."""
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(mask)))
    return segments


# ─────────────────────────────────────── evaluation runner ─────────────────────

def evaluate_method(
    series: np.ndarray,
    gt_indices: np.ndarray,
    method: str,
    dataset_name: str,
    ae_epochs: int = 15,
) -> dict:
    n = len(series)
    y_true = _binary_label_array(gt_indices, n)

    t0 = time.time()
    result = detect_anomalies(
        series,
        method=method,
        dataset_name=dataset_name,
        save_plot=False,
        ae_epochs=ae_epochs,
    )
    elapsed = time.time() - t0

    y_pred = _binary_label_array(result["anomaly_indices"], n)
    scores = result["scores"]

    metrics = precision_recall_f1(y_true, y_pred)
    auc = roc_auc(y_true, scores)
    pa_f1 = point_adjust_f1(y_true, y_pred, window=15)

    return {
        "precision":      round(metrics["precision"], 4),
        "recall":         round(metrics["recall"], 4),
        "f1":             round(metrics["f1"], 4),
        "roc_auc":        round(auc, 4),
        "point_adj_f1":   round(pa_f1, 4),
        "n_detected":     len(result["anomaly_indices"]),
        "n_gt":           int(y_true.sum()),
        "elapsed_sec":    round(elapsed, 2),
    }


# ─────────────────────────────────────── visualization ─────────────────────────

def plot_comparison(summary: dict, out_dir: str = OUT_DIR) -> None:
    """Bar chart comparing F1 / ROC-AUC across method × dataset."""
    datasets = list(summary.keys())
    methods_all: list[str] = []
    for ds_metrics in summary.values():
        for m in ds_metrics:
            if m not in methods_all:
                methods_all.append(m)

    x = np.arange(len(datasets))
    width = 0.8 / len(methods_all)
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["f1", "roc_auc"]):
        for j, method in enumerate(methods_all):
            vals = [summary[ds].get(method, {}).get(metric, 0.0) for ds in datasets]
            offsets = x + (j - len(methods_all) / 2 + 0.5) * width
            bars = ax.bar(offsets, vals, width=width * 0.9,
                          label=method, color=colors[j % len(colors)])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Anomaly Detection — {metric.upper()}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "evaluation_comparison.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved → {out}")


# ─────────────────────────────────────── main ──────────────────────────────────

def main(ae_epochs: int = 15):
    generators = {
        "ecg":        generate_ecg,
        "industrial": generate_industrial,
        "financial":  generate_financial,
    }
    methods = ["autoencoder", "isolation_forest", "prophet"]

    summary: dict[str, dict[str, dict]] = {}
    print("\n" + "=" * 70)
    print("Time Series Anomaly Detection — Full Benchmark")
    print("=" * 70)

    for ds_name, gen_fn in generators.items():
        print(f"\n[Dataset: {ds_name}]")
        data, gt_idx = gen_fn()
        summary[ds_name] = {}

        for method in methods:
            print(f"  {method:20s}", end=" ", flush=True)
            metrics = evaluate_method(
                data, gt_idx, method, ds_name, ae_epochs=ae_epochs
            )
            summary[ds_name][method] = metrics
            print(f"F1={metrics['f1']:.3f}  AUC={metrics['roc_auc']:.3f}"
                  f"  PA-F1={metrics['point_adj_f1']:.3f}"
                  f"  ({metrics['elapsed_sec']:.1f}s)")

    # Save JSON
    out_json = os.path.join(OUT_DIR, "results_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Plot
    plot_comparison(summary)

    # Print table
    print("\n" + "─" * 70)
    print(f"{'Dataset':12s} {'Method':22s} {'F1':>6} {'AUC':>6} {'PA-F1':>7}")
    print("─" * 70)
    for ds, methods_res in summary.items():
        for method, m in methods_res.items():
            print(f"{ds:12s} {method:22s} {m['f1']:6.3f} {m['roc_auc']:6.3f} {m['point_adj_f1']:7.3f}")
    print("─" * 70)

    return summary


if __name__ == "__main__":
    main(ae_epochs=15)
