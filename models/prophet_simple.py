"""
models/prophet_simple.py — Simplified trend + seasonality decomposition model.

Steps
-----
1. Trend    : centered moving average
2. Seasonal : FFT-based dominant period → reconstruct seasonal component
3. Residual : original − trend − seasonal
4. Anomaly  : |residual| > k × IQR(residual)
"""
from __future__ import annotations

import numpy as np


class ProphetSimple:
    """
    Decomposition-based anomaly detector (no external dependencies).

    Parameters
    ----------
    trend_window : window for centered moving average trend estimate
    k_iqr       : anomaly threshold multiplier on IQR of residuals
    max_periods : number of dominant FFT frequencies to reconstruct seasonal
    """

    def __init__(self, trend_window: int = 24, k_iqr: float = 3.0,
                 max_periods: int = 3):
        self.trend_window = trend_window
        self.k_iqr = k_iqr
        self.max_periods = max_periods

        # Fitted components
        self.trend_: np.ndarray | None = None
        self.seasonal_: np.ndarray | None = None
        self.residual_: np.ndarray | None = None
        self.threshold_low_: float = 0.0
        self.threshold_high_: float = 0.0
        self._n_fit: int = 0

    # ─────────────────────────────────────────────────────── fit
    def fit(self, series: np.ndarray) -> "ProphetSimple":
        """Fit decomposition on training series."""
        series = np.asarray(series, dtype=float)
        self._n_fit = len(series)
        trend = self._moving_average(series, self.trend_window)
        detrended = series - trend
        seasonal = self._fft_seasonal(detrended, self.max_periods)
        residual = detrended - seasonal

        self.trend_ = trend
        self.seasonal_ = seasonal
        self.residual_ = residual

        q25, q75 = np.nanpercentile(residual, [25, 75])
        iqr = q75 - q25
        self.threshold_low_  = float(np.nanmedian(residual) - self.k_iqr * iqr)
        self.threshold_high_ = float(np.nanmedian(residual) + self.k_iqr * iqr)
        return self

    # ─────────────────────────────────────────────────────── predict
    def predict(self, series: np.ndarray) -> dict:
        """
        Decompose `series` and return per-point residuals + anomaly flags.
        If `series` length differs from fit length, seasonal is re-estimated.
        """
        series = np.asarray(series, dtype=float)
        trend = self._moving_average(series, self.trend_window)
        detrended = series - trend
        seasonal = self._fft_seasonal(detrended, self.max_periods)
        residual = detrended - seasonal

        anomaly_mask = (residual < self.threshold_low_) | (residual > self.threshold_high_)
        anomaly_indices = np.where(anomaly_mask)[0].tolist()

        # Build continuous anomaly score: normalised |residual|
        span = max(abs(self.threshold_high_), abs(self.threshold_low_), 1e-9)
        scores = np.abs(residual) / span

        return {
            "anomaly_indices": anomaly_indices,
            "scores": scores,
            "threshold": float(self.threshold_high_),
            "residual": residual,
            "trend": trend,
            "seasonal": seasonal,
        }

    def get_anomalies(self, series: np.ndarray) -> np.ndarray:
        """Convenience: return anomaly index array."""
        return np.array(self.predict(series)["anomaly_indices"], dtype=int)

    # ─────────────────────────────────────────────────────── decomposition helpers
    @staticmethod
    def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
        """Centered moving average; edges padded with nearest valid value."""
        if window < 2:
            return x.copy()
        w = window if window % 2 == 1 else window + 1   # ensure odd
        half = w // 2
        n = len(x)
        trend = np.full(n, np.nan)

        kernel = np.ones(w) / w
        # Use numpy convolve on valid region
        valid = np.convolve(x, kernel, mode="valid")
        trend[half: half + len(valid)] = valid

        # Fill edges by repeating boundary values
        trend[:half] = trend[half]
        trend[half + len(valid):] = trend[half + len(valid) - 1]
        return trend

    @staticmethod
    def _fft_seasonal(detrended: np.ndarray, max_periods: int = 3) -> np.ndarray:
        """
        Estimate seasonal component by keeping the `max_periods` dominant
        FFT frequencies (excluding DC and the very low-freq trend residual).
        """
        n = len(detrended)
        # Mask NaN for FFT
        mask = np.isnan(detrended)
        x = detrended.copy()
        x[mask] = 0.0

        fft_vals = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n)

        # Ignore DC (0) and very-low-frequency components
        magnitudes = np.abs(fft_vals)
        magnitudes[0] = 0.0   # remove DC
        # Ignore frequencies corresponding to periods > n/4
        min_freq = 4.0 / n
        magnitudes[freqs < min_freq] = 0.0

        # Find top-k frequencies
        top_idx = np.argsort(magnitudes)[::-1][:max_periods]

        # Reconstruct using only those frequencies
        fft_filtered = np.zeros_like(fft_vals)
        fft_filtered[top_idx] = fft_vals[top_idx]
        seasonal = np.fft.irfft(fft_filtered, n=n)
        seasonal[mask] = 0.0
        return seasonal


# ─────────────────────────────────────── main ──────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from generate_data import generate_industrial

    print("Testing ProphetSimple on Industrial sensor data …")
    data, gt_idx = generate_industrial()

    model = ProphetSimple(trend_window=24, k_iqr=3.0, max_periods=3)
    model.fit(data[:7000])
    result = model.predict(data)

    print(f"Threshold (±)       : {result['threshold']:.4f}")
    print(f"Detected anomalies  : {len(result['anomaly_indices'])}")
    print(f"Ground truth        : {len(gt_idx)}")
