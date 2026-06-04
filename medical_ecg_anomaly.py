"""
medical_ecg_anomaly.py — Medical ECG Anomaly Detection

Applies all 5 anomaly detectors to ECG-specific use case.
Each anomaly type represents a clinical event:
  - Spike anomaly  → possible artifact or ectopic beat
  - Level shift    → possible electrode displacement (flat segment = 'missing beat')
  - Missing segment→ lead dropout (flat, near-zero values)

Evaluates per anomaly-type sensitivity/specificity and provides
clinical interpretation of which detector best matches which event type.

Outputs:
  medical_ecg_benchmark.png  — per-type performance heatmap
  clinical_interpretation.json — interpretation results
"""
from __future__ import annotations

import sys
import os
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.dirname(__file__)


# ─────────────────────────────────────── helpers ───────────────────────────────

def roc_auc_manual(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn."""
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.5
    pos_scores = scores[pos_idx]
    neg_scores = scores[neg_idx]
    count = 0.0
    for ps in pos_scores:
        count += float(np.sum(ps > neg_scores))
        count += 0.5 * float(np.sum(ps == neg_scores))
    return count / (len(pos_scores) * len(neg_scores))


def compute_metrics_binary(pred_indices: list[int], scores: np.ndarray,
                            gt_indices: np.ndarray, n: int) -> dict[str, float]:
    gt_set   = set(gt_indices.tolist())
    pred_set = set(pred_indices)
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    tn = n - tp - fp - fn

    precision   = tp / (tp + fp + 1e-12)
    recall      = tp / (tp + fn + 1e-12)   # sensitivity
    specificity = tn / (tn + fp + 1e-12)
    f1          = 2 * precision * recall / (precision + recall + 1e-12)

    labels = np.zeros(n, dtype=np.int32)
    labels[list(gt_set)] = 1
    s_min, s_max = scores.min(), scores.max()
    norm_scores  = (scores - s_min) / (s_max - s_min + 1e-12)
    auc = roc_auc_manual(norm_scores, labels)

    return {
        "precision":   round(float(precision), 4),
        "recall":      round(float(recall), 4),      # sensitivity
        "specificity": round(float(specificity), 4),
        "f1":          round(float(f1), 4),
        "auc":         round(float(auc), 4),
    }


# ─────────────────────────────────────── ECG data generation ───────────────────

def generate_ecg_typed(n: int = 10_000, seed: int = 42):
    """
    Generate ECG signal with explicitly typed anomaly indices:
      spike_idx : ectopic beats / artifacts (amplitude spikes)
      flat_idx  : lead dropout (flat near-zero segments)

    Returns:
      data          : ECG signal (np.ndarray)
      spike_idx     : indices of spike anomalies
      flat_idx      : indices of flat (missing segment) anomalies
      all_anomalies : combined anomaly indices
    """
    from generate_data import _inject_spikes, _choose_anomaly_indices

    rng = np.random.default_rng(seed)
    ANOMALY_RATE = 0.02
    t = np.arange(n)

    freq = 1 / 143
    base = np.sin(2 * np.pi * freq * t)
    base += 0.3 * np.sin(4 * np.pi * freq * t)
    base += 0.05 * rng.standard_normal(n)

    n_anomalies = int(n * ANOMALY_RATE)
    spike_k = n_anomalies // 2
    flat_k  = n_anomalies - spike_k

    spike_idx = _choose_anomaly_indices(rng, n, spike_k / n)
    flat_idx  = _choose_anomaly_indices(rng, n, flat_k / n, exclude=set(spike_idx))

    data = _inject_spikes(rng, base, spike_idx, magnitude=4.0)

    for i in flat_idx:
        end = min(n, i + 15)
        data[i:end] = rng.uniform(-0.05, 0.05, end - i)

    # Expand flat anomaly to cover the full 15-sample window
    flat_expanded: list[int] = []
    for i in flat_idx:
        flat_expanded.extend(range(i, min(n, i + 15)))
    flat_expanded_arr = np.array(sorted(set(flat_expanded)), dtype=np.int64)

    spike_arr = np.array(sorted(spike_idx), dtype=np.int64)
    all_anom  = np.array(sorted(set(spike_idx) | set(flat_expanded)), dtype=np.int64)

    return data, spike_arr, flat_expanded_arr, all_anom


# ─────────────────────────────────────── method runners ───────────────────────

def _run_detector(name: str, data: np.ndarray, gt_all: np.ndarray, n: int) -> dict:
    """Run a single detector; return dict with pred_indices and scores."""
    split = int(len(data) * 0.70)
    train = data[:split]

    try:
        if name == "Prophet":
            from models.prophet_simple import ProphetSimple
            model = ProphetSimple(trend_window=24, k_iqr=3.0, max_periods=3)
            model.fit(train)
            res = model.predict(data)
        elif name == "IsolationForest":
            from models.isolation_forest import IsolationForestDetector
            model = IsolationForestDetector(window=50, step=10, contamination=0.02)
            model.fit(train)
            res = model.predict(data)
        elif name == "LSTM_AE":
            from models.autoencoder import AnomalyAutoencoder
            model = AnomalyAutoencoder(window=50, hidden=32, latent=16, epochs=10, batch_size=128)
            model.fit(train, verbose=False)
            res = model.predict(data)
        elif name == "TCN":
            from models.tcn_anomaly import TCNAnomalyDetector
            model = TCNAnomalyDetector(window=50, stride=25, enc_channels=32,
                                       bottleneck=16, epochs=10, batch_size=128)
            model.fit(train, verbose=False)
            res = model.predict(data)
        elif name == "LSTM_VAE":
            from models.lstm_vae import LSTMVAEDetector
            model = LSTMVAEDetector(window=50, stride=25, hidden_size=64,
                                    latent_dim=32, beta=0.1, epochs=10, batch_size=256)
            model.fit(train, verbose=False)
            res = model.predict(data)
        else:
            raise ValueError(f"Unknown detector: {name}")

        return {"pred_indices": res["anomaly_indices"], "scores": res["scores"]}
    except Exception as exc:
        print(f"    ERROR in {name}: {exc}")
        scores = np.zeros(n)
        return {"pred_indices": [], "scores": scores}


# ─────────────────────────────────────── per-type evaluation ───────────────────

def evaluate_per_type(pred_indices: list[int], scores: np.ndarray,
                       spike_gt: np.ndarray, flat_gt: np.ndarray, n: int) -> dict:
    """Evaluate detector performance per anomaly type."""
    spike_metrics = compute_metrics_binary(pred_indices, scores, spike_gt, n)
    flat_metrics  = compute_metrics_binary(pred_indices, scores, flat_gt, n)
    all_gt        = np.concatenate([spike_gt, flat_gt])
    all_metrics   = compute_metrics_binary(pred_indices, scores, all_gt, n)
    return {
        "spike": spike_metrics,
        "flat":  flat_metrics,
        "all":   all_metrics,
    }


# ─────────────────────────────────────── visualization ────────────────────────

def plot_results(results: dict, save_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        methods = list(results.keys())
        types   = ["spike", "flat", "all"]
        metrics = ["recall", "specificity", "f1", "auc"]

        n_methods = len(methods)
        n_types   = len(types)
        n_metrics = len(metrics)

        fig, axes = plt.subplots(n_types, n_metrics, figsize=(4 * n_metrics, 3.5 * n_types),
                                  squeeze=False)
        fig.suptitle("Medical ECG Anomaly Detection — Per-Type Performance",
                     fontsize=14, fontweight="bold")

        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
        type_labels = {
            "spike": "Spike (ectopic/artifact)",
            "flat":  "Flat (lead dropout)",
            "all":   "All anomalies",
        }
        metric_labels = {
            "recall":      "Sensitivity (Recall)",
            "specificity": "Specificity",
            "f1":          "F1 Score",
            "auc":         "ROC-AUC",
        }

        for ti, atype in enumerate(types):
            for mi, metric in enumerate(metrics):
                ax = axes[ti][mi]
                vals = [results[m][atype].get(metric, 0.0) for m in methods]
                bars = ax.bar(methods, vals, color=colors, edgecolor="white", linewidth=0.8)
                ax.set_title(f"{type_labels[atype]}\n{metric_labels[metric]}", fontsize=8)
                ax.set_ylim(0, 1.1)
                ax.set_xticks(range(len(methods)))
                ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=7)
                ax.yaxis.set_tick_params(labelsize=7)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            val + 0.02, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=6)
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")
    except Exception as exc:
        print(f"  Plot skipped: {exc}")


# ─────────────────────────────────────── clinical interpretation ───────────────

def generate_clinical_interpretation(results: dict) -> dict:
    """
    Determine which detector best suits each clinical event type.
    Best = highest F1 for that anomaly type.
    """
    methods = list(results.keys())

    interpretation: dict = {}

    # Best detector per type (by F1)
    for atype in ["spike", "flat", "all"]:
        best_method = max(methods, key=lambda m: results[m][atype]["f1"])
        best_f1     = results[best_method][atype]["f1"]
        best_recall = results[best_method][atype]["recall"]
        best_spec   = results[best_method][atype]["specificity"]

        interpretation[atype] = {
            "best_detector": best_method,
            "best_f1": best_f1,
            "best_sensitivity": best_recall,
            "best_specificity": best_spec,
        }

    # Clinical commentary
    spike_best   = interpretation["spike"]["best_detector"]
    flat_best    = interpretation["flat"]["best_detector"]
    overall_best = interpretation["all"]["best_detector"]

    clinical_notes = {
        "ectopic_beat_detection": {
            "clinical_event": "Ectopic beat / artifact (spike anomaly)",
            "recommended_detector": spike_best,
            "rationale": (
                f"{spike_best} shows highest F1 for spike anomalies "
                "(sudden, isolated amplitude excursions). "
                "These events are brief (1-2 samples) and high-amplitude — "
                "threshold-based methods tend to work well."
            ),
            "clinical_action": (
                "Sudden amplitude spikes in ECG should trigger review for: "
                "(1) Ectopic/premature beats — evaluate for arrhythmia if frequent. "
                "(2) Movement artifact — verify lead placement, ask patient to remain still."
            ),
        },
        "electrode_displacement": {
            "clinical_event": "Lead dropout / flat segment (missing beat anomaly)",
            "recommended_detector": flat_best,
            "rationale": (
                f"{flat_best} shows highest F1 for flat-segment anomalies "
                "(sustained near-zero signal). "
                "These are multi-sample events — reconstruction-based methods "
                "(LSTM_AE / LSTM_VAE) can detect the pattern change over a window."
            ),
            "clinical_action": (
                "Flat ECG segments suggest: "
                "(1) Lead dropout — re-attach electrode and verify contact quality. "
                "(2) True asystole — if unexpected, immediately initiate resuscitation protocol. "
                "(3) Electrode displacement — systematic lead check required."
            ),
        },
        "overall_monitoring": {
            "clinical_event": "General ECG monitoring (all anomaly types)",
            "recommended_detector": overall_best,
            "rationale": (
                f"{overall_best} provides the best overall F1 across all anomaly types. "
                "For continuous bedside monitoring where both artifacts and clinical events "
                "must be detected, a combined ensemble approach is preferred."
            ),
            "clinical_action": (
                "For continuous ICU/ward ECG monitoring: "
                "Layer 1 (fast): Rule-based threshold (Prophet/IQR) for immediate spike alerts. "
                "Layer 2 (precise): LSTM-based model for sustained pattern changes. "
                "False positive rate management is critical — alert fatigue reduces clinical utility."
            ),
        },
    }

    # Summary table
    summary = {
        "per_type_best": {
            "spike": interpretation["spike"],
            "flat":  interpretation["flat"],
            "all":   interpretation["all"],
        },
        "clinical_notes": clinical_notes,
        "detector_rankings": {},
    }

    # Rank all detectors by overall F1
    ranked = sorted(methods, key=lambda m: results[m]["all"]["f1"], reverse=True)
    for rank, m in enumerate(ranked, 1):
        summary["detector_rankings"][m] = {
            "overall_rank": rank,
            "all_f1":  results[m]["all"]["f1"],
            "spike_f1": results[m]["spike"]["f1"],
            "flat_f1":  results[m]["flat"]["f1"],
            "auc":      results[m]["all"]["auc"],
        }

    return summary


# ─────────────────────────────────────── main ──────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Medical ECG Anomaly Detection Benchmark")
    print("=" * 65)

    # Generate typed ECG data
    print("\nGenerating typed ECG data …")
    data, spike_idx, flat_idx, all_idx = generate_ecg_typed(n=10_000, seed=42)
    n = len(data)
    print(f"  Total points : {n:,}")
    print(f"  Spike anomalies : {len(spike_idx)}")
    print(f"  Flat anomalies  : {len(flat_idx)}")
    print(f"  All anomalies   : {len(all_idx)} ({len(all_idx)/n*100:.1f}%)")

    METHODS = ["Prophet", "IsolationForest", "LSTM_AE", "TCN", "LSTM_VAE"]

    all_results: dict[str, dict] = {}

    for method in METHODS:
        print(f"\n  Running {method} …", end=" ", flush=True)
        t0 = time.time()
        det_output = _run_detector(method, data, all_idx, n)
        elapsed = time.time() - t0
        per_type = evaluate_per_type(
            det_output["pred_indices"], det_output["scores"],
            spike_idx, flat_idx, n
        )
        per_type["time_s"] = round(elapsed, 2)
        all_results[method] = per_type

        print(f"done ({elapsed:.1f}s)")
        print(f"    Spike : sensitivity={per_type['spike']['recall']:.3f}"
              f"  specificity={per_type['spike']['specificity']:.3f}"
              f"  F1={per_type['spike']['f1']:.3f}")
        print(f"    Flat  : sensitivity={per_type['flat']['recall']:.3f}"
              f"  specificity={per_type['flat']['specificity']:.3f}"
              f"  F1={per_type['flat']['f1']:.3f}")
        print(f"    All   : F1={per_type['all']['f1']:.3f}"
              f"  AUC={per_type['all']['auc']:.3f}")

    # Print summary table
    print("\n" + "=" * 65)
    print("Summary Table — ECG Anomaly Detection per Type")
    print("=" * 65)
    hdr = f"{'Method':<18}  {'Spike F1':>9}  {'Flat F1':>8}  {'All F1':>7}  {'AUC':>7}"
    print(hdr)
    print("-" * len(hdr))
    for m in METHODS:
        r = all_results[m]
        print(f"  {m:<16}  {r['spike']['f1']:>9.4f}"
              f"  {r['flat']['f1']:>8.4f}"
              f"  {r['all']['f1']:>7.4f}"
              f"  {r['all']['auc']:>7.4f}")

    # Generate clinical interpretation
    interpretation = generate_clinical_interpretation(all_results)

    # Save outputs
    plot_path = os.path.join(BASE_DIR, "medical_ecg_benchmark.png")
    plot_results(all_results, plot_path)

    interp_path = os.path.join(BASE_DIR, "clinical_interpretation.json")
    with open(interp_path, "w", encoding="utf-8") as f:
        json.dump(interpretation, f, ensure_ascii=False, indent=2)
    print(f"\nClinical interpretation saved to: {interp_path}")

    print("\n=== Clinical Interpretation ===")
    for event, note in interpretation["clinical_notes"].items():
        print(f"\n[{note['clinical_event']}]")
        print(f"  Best detector : {note['recommended_detector']}")
        print(f"  Rationale     : {note['rationale'][:120]}…")

    print("\n=== Detector Rankings (by overall F1) ===")
    for m, rank_info in interpretation["detector_rankings"].items():
        print(f"  #{rank_info['overall_rank']} {m:<18} "
              f"F1_all={rank_info['all_f1']:.4f}  "
              f"F1_spike={rank_info['spike_f1']:.4f}  "
              f"F1_flat={rank_info['flat_f1']:.4f}")
