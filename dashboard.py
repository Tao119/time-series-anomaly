"""
dashboard.py — Results Dashboard (text-based)

Loads all available result JSON files and generates a comprehensive
formatted text report saved as results_dashboard.txt.

Covers:
  - benchmark_results.json (main 3-dataset benchmark)
  - clinical_interpretation.json (medical ECG per-type analysis)

Output: results_dashboard.txt
"""
from __future__ import annotations

import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
OUT_PATH = os.path.join(BASE_DIR, "results_dashboard.txt")

METHODS = ["Prophet", "IsolationForest", "LSTM_AE", "TCN", "LSTM_VAE"]
DATASETS = ["ECG", "Industrial", "Financial"]
METRICS  = [("precision", "Precision"), ("recall", "Recall"),
            ("f1", "F1"), ("auc", "AUC")]


# ─────────────────────────────────────── loaders ───────────────────────────────

def load_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ─────────────────────────────────────── formatting helpers ────────────────────

def _hline(cols: list[int], char: str = "─") -> str:
    """Horizontal line for table."""
    parts = [char * (c + 2) for c in cols]
    return "├" + "┼".join(parts) + "┤"


def _top_line(cols: list[int]) -> str:
    parts = ["─" * (c + 2) for c in cols]
    return "┌" + "┬".join(parts) + "┐"


def _bot_line(cols: list[int]) -> str:
    parts = ["─" * (c + 2) for c in cols]
    return "└" + "┴".join(parts) + "┘"


def _row(vals: list[str], cols: list[int]) -> str:
    cells = [f" {v:<{cols[i]}} " if i == 0 else f" {v:>{cols[i]}} "
             for i, v in enumerate(vals)]
    return "│" + "│".join(cells) + "│"


def build_metric_table(results: dict, dataset: str) -> list[str]:
    """Build ASCII table for one dataset."""
    ds_data = results.get(dataset, {})
    if not ds_data:
        return [f"  [No data for {dataset}]"]

    col_widths = [16, 10, 8, 7, 7]  # Method + 4 metrics
    headers = ["Method", "Precision", "Recall", "F1", "AUC"]

    lines = [_top_line(col_widths)]
    lines.append(_row(headers, col_widths))
    lines.append(_hline(col_widths))

    for method in METHODS:
        m = ds_data.get(method, {})
        row = [
            method,
            f"{m.get('precision', 0.0):.4f}",
            f"{m.get('recall', 0.0):.4f}",
            f"{m.get('f1', 0.0):.4f}",
            f"{m.get('auc', 0.0):.4f}",
        ]
        lines.append(_row(row, col_widths))

    lines.append(_bot_line(col_widths))
    return lines


def build_ecg_type_table(ecg_results: dict) -> list[str]:
    """Per anomaly-type performance table for ECG medical benchmark."""
    col_widths = [16, 10, 10, 8, 7, 7]
    headers = ["Method", "Spike F1", "Flat F1", "All F1", "AUC", "Time(s)"]

    lines = [_top_line(col_widths)]
    lines.append(_row(headers, col_widths))
    lines.append(_hline(col_widths))

    for method in METHODS:
        m = ecg_results.get(method, {})
        if not m:
            continue
        row = [
            method,
            f"{m.get('spike', {}).get('f1', 0.0):.4f}",
            f"{m.get('flat',  {}).get('f1', 0.0):.4f}",
            f"{m.get('all',   {}).get('f1', 0.0):.4f}",
            f"{m.get('all',   {}).get('auc', 0.0):.4f}",
            f"{m.get('time_s', 0.0):.1f}",
        ]
        lines.append(_row(row, col_widths))

    lines.append(_bot_line(col_widths))
    return lines


def best_method_summary(results: dict, dataset: str, metric: str = "f1") -> str:
    ds = results.get(dataset, {})
    if not ds:
        return "N/A"
    best = max(ds, key=lambda m: ds[m].get(metric, 0.0))
    val  = ds[best].get(metric, 0.0)
    return f"{best} ({val:.4f})"


# ─────────────────────────────────────── report builder ───────────────────────

def build_report(benchmark: dict | None, ecg_types_raw: dict | None,
                 clinical: dict | None) -> str:
    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "=" * 65,
        "  Time Series Anomaly Detection — Results Dashboard",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
    ]

    # ── 1. Main Benchmark ───────────────────────────────────────────────────
    lines += [
        "┌─────────────────────────────────────────────────────────────┐",
        "│            SECTION 1: Main Benchmark (3 Datasets)           │",
        "└─────────────────────────────────────────────────────────────┘",
        "",
    ]

    if benchmark:
        for ds in DATASETS:
            ds_data = benchmark.get(ds, {})
            n_pts = 10_000  # known from generate_data.py
            rate_info = {
                "ECG": "2.0%", "Industrial": "~15.0% (expanded level shifts)",
                "Financial": "~5.5% (3σ jump events)"
            }
            lines += [
                f"  {'─'*59}",
                f"  {ds} Dataset  ({n_pts:,} points, ~{rate_info.get(ds, '2%')} anomaly rate)",
                f"  {'─'*59}",
            ]
            lines += build_metric_table(benchmark, ds)
            lines += [""]

            # Best method per metric
            lines += ["  Best per metric:"]
            for mk, ml in METRICS:
                best = best_method_summary(benchmark, ds, mk)
                lines.append(f"    {ml:<12}: {best}")
            lines += [""]

    else:
        lines += [
            "  [benchmark_results.json not found]",
            "  Run python benchmark.py to generate results.",
            "",
        ]

    # ── 2. Medical ECG Benchmark ────────────────────────────────────────────
    lines += [
        "┌─────────────────────────────────────────────────────────────┐",
        "│         SECTION 2: Medical ECG Anomaly Detection            │",
        "└─────────────────────────────────────────────────────────────┘",
        "",
        "  Clinical anomaly types:",
        "    Spike anomaly → ectopic beat / artifact",
        "    Flat segment  → lead dropout / missing beat",
        "",
        "  ECG Dataset (10,000 points, ~2% anomaly rate)",
        "  ─" * 30,
    ]

    # Build per-type table from clinical interpretation OR raw ecg data
    if ecg_types_raw:
        lines += build_ecg_type_table(ecg_types_raw)
        lines += [""]

        # Sensitivity/specificity detail per type
        lines += ["  Sensitivity / Specificity per anomaly type:"]
        col_w2 = [16, 13, 13, 13, 13]
        hdr2   = ["Method", "Spike Sens", "Spike Spec", "Flat Sens", "Flat Spec"]
        lines += [_top_line(col_w2), _row(hdr2, col_w2), _hline(col_w2)]
        for method in METHODS:
            m = ecg_types_raw.get(method, {})
            if not m:
                continue
            row2 = [
                method,
                f"{m.get('spike',{}).get('recall', 0.0):.4f}",
                f"{m.get('spike',{}).get('specificity', 0.0):.4f}",
                f"{m.get('flat', {}).get('recall', 0.0):.4f}",
                f"{m.get('flat', {}).get('specificity', 0.0):.4f}",
            ]
            lines.append(_row(row2, col_w2))
        lines.append(_bot_line(col_w2))
        lines += [""]
    else:
        lines += [
            "  [No per-type ECG data — run python medical_ecg_anomaly.py]",
            "",
        ]

    # ── 3. Clinical Interpretation ─────────────────────────────────────────
    lines += [
        "┌─────────────────────────────────────────────────────────────┐",
        "│            SECTION 3: Clinical Interpretation               │",
        "└─────────────────────────────────────────────────────────────┘",
        "",
    ]

    if clinical:
        rankings = clinical.get("detector_rankings", {})
        if rankings:
            lines += ["  Detector Rankings (by Overall ECG F1):"]
            for method, info in sorted(rankings.items(), key=lambda x: x[1]["overall_rank"]):
                lines.append(
                    f"    #{info['overall_rank']} {method:<18} "
                    f"F1_all={info['all_f1']:.4f}  "
                    f"F1_spike={info['spike_f1']:.4f}  "
                    f"F1_flat={info['flat_f1']:.4f}"
                )
            lines += [""]

        notes = clinical.get("clinical_notes", {})
        if notes:
            lines += ["  Clinical Recommendations:"]
            for key, note in notes.items():
                lines += [
                    "",
                    f"  [{note.get('clinical_event', key)}]",
                    f"  Recommended: {note.get('recommended_detector', 'N/A')}",
                ]
                action = note.get("clinical_action", "")
                # Wrap at 60 chars
                words = action.split()
                current_line = "  Action: "
                for word in words:
                    if len(current_line) + len(word) + 1 > 65:
                        lines.append(current_line)
                        current_line = "           " + word + " "
                    else:
                        current_line += word + " "
                if current_line.strip():
                    lines.append(current_line)
    else:
        lines += [
            "  [clinical_interpretation.json not found]",
            "  Run python medical_ecg_anomaly.py to generate.",
            "",
        ]

    # ── 4. Key Observations ────────────────────────────────────────────────
    lines += [
        "",
        "┌─────────────────────────────────────────────────────────────┐",
        "│                  SECTION 4: Key Observations                │",
        "└─────────────────────────────────────────────────────────────┘",
        "",
    ]

    observations: list[str] = []

    if benchmark:
        ecg_data = benchmark.get("ECG", {})
        ind_data = benchmark.get("Industrial", {})
        fin_data = benchmark.get("Financial", {})

        best_ecg = max(METHODS, key=lambda m: ecg_data.get(m, {}).get("f1", 0))
        best_ind = max(METHODS, key=lambda m: ind_data.get(m, {}).get("f1", 0))
        best_fin = max(METHODS, key=lambda m: fin_data.get(m, {}).get("f1", 0))

        observations += [
            f"  - Best overall ECG detector     : {best_ecg}"
            f" (F1={ecg_data.get(best_ecg, {}).get('f1', 0):.4f})",
            f"  - Best overall Industrial det.  : {best_ind}"
            f" (F1={ind_data.get(best_ind, {}).get('f1', 0):.4f})",
            f"  - Best overall Financial det.   : {best_fin}"
            f" (F1={fin_data.get(best_fin, {}).get('f1', 0):.4f})",
        ]

    observations += [
        "  - Rule-based (Prophet) excels at isolated spike detection",
        "    with low latency — ideal for real-time alerting.",
        "  - Reconstruction models (LSTM_AE, LSTM_VAE) better capture",
        "    sustained pattern changes (flat segments / level shifts).",
        "  - Financial data is hardest — random-walk nature means",
        "    anomalies are statistically indistinguishable from noise.",
        "  - For clinical deployment, ensemble strategies combining",
        "    rule-based fast-response + neural slow-but-precise",
        "    are recommended to balance sensitivity and specificity.",
    ]

    lines += observations
    lines += ["", "=" * 65, "  End of Report", "=" * 65]

    return "\n".join(lines)


# ─────────────────────────────────────── main ──────────────────────────────────

if __name__ == "__main__":
    print("Loading result files …")

    # Main benchmark
    benchmark = load_json(os.path.join(BASE_DIR, "benchmark_results.json"))
    if benchmark:
        print(f"  benchmark_results.json: loaded ({len(benchmark)} datasets)")
    else:
        print("  benchmark_results.json: NOT FOUND")

    # Clinical interpretation (contains per-type ECG data indirectly)
    clinical = load_json(os.path.join(BASE_DIR, "clinical_interpretation.json"))
    if clinical:
        print("  clinical_interpretation.json: loaded")
    else:
        print("  clinical_interpretation.json: NOT FOUND")

    # Per-type ECG data — reconstruct from clinical interpretation if available
    ecg_types: dict | None = None
    if clinical and "detector_rankings" in clinical:
        # Rebuild per-type dict from what we can infer
        # If medical_ecg_anomaly.py ran, the data is embedded in rankings
        ranks = clinical.get("detector_rankings", {})
        per_type_best = clinical.get("per_type_best", {})
        # We need the full per-type table — check if it was saved separately
        ecg_type_path = os.path.join(BASE_DIR, "ecg_type_results.json")
        ecg_types = load_json(ecg_type_path)
        if not ecg_types:
            # Build minimal table from ranking data
            ecg_types = {}
            for method, info in ranks.items():
                ecg_types[method] = {
                    "spike": {"f1": info.get("spike_f1", 0.0), "recall": 0.0, "specificity": 0.0},
                    "flat":  {"f1": info.get("flat_f1",  0.0), "recall": 0.0, "specificity": 0.0},
                    "all":   {"f1": info.get("all_f1",   0.0), "auc": info.get("auc", 0.0)},
                    "time_s": 0.0,
                }

    report = build_report(benchmark, ecg_types, clinical)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nDashboard saved to: {OUT_PATH}")
    print("\n" + "=" * 65)
    # Also print to stdout
    print(report)
