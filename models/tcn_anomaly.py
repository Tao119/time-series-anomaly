"""
models/tcn_anomaly.py — Temporal Convolutional Network (TCN) for Anomaly Detection

Architecture
------------
  5 TCN blocks with dilations [1, 2, 4, 8, 16]
  Each block: dilated causal conv → LayerNorm → ReLU → residual connection
  Global average pooling → Dense head

Two modes:
  1. Classification: detect anomaly windows (supervised, binary label)
  2. Reconstruction: autoencoder — anomaly = high reconstruction error

Window size: 50, stride: 25
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────── TCN building blocks ───────────────────

class CausalConv1d(nn.Module):
    """Causal 1D convolution with dilation (no future information leakage)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length)
        out = self.conv(x)
        # Remove future-facing padding
        return out[:, :, :x.shape[2]]


class TCNBlock(nn.Module):
    """
    Single TCN residual block:
      causal conv → LayerNorm → ReLU → causal conv → LayerNorm → ReLU
      + residual connection (with 1×1 conv if channels differ)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual projection if channel mismatch
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)                         # (B, C, L)
        out = self.norm1(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.relu(out)
        out = self.dropout(out)

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return self.relu(out + residual)


# ─────────────────────────────────────── TCN models ────────────────────────────

class TCNClassifier(nn.Module):
    """
    TCN for supervised anomaly window classification.
    Input:  (batch, seq_len, 1)  → reshaped to (batch, 1, seq_len)
    Output: (batch, 1) — anomaly probability
    """

    def __init__(self, in_channels: int = 1, num_channels: int = 32,
                 kernel_size: int = 3, dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
                 dropout: float = 0.1):
        super().__init__()
        channels = [in_channels] + [num_channels] * len(dilations)
        blocks = []
        for i, d in enumerate(dilations):
            blocks.append(TCNBlock(channels[i], channels[i + 1],
                                   kernel_size, d, dropout))
        self.tcn = nn.Sequential(*blocks)
        self.classifier = nn.Linear(num_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1) → (B, 1, L)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)                           # (B, C, L)
        pooled = out.mean(dim=2)                     # (B, C)
        return self.classifier(pooled)               # (B, 1)


class TCNAutoencoder(nn.Module):
    """
    TCN Autoencoder for unsupervised anomaly detection.
    Encoder: TCN blocks → bottleneck
    Decoder: TCN blocks → reconstruct sequence
    """

    def __init__(self, in_channels: int = 1, enc_channels: int = 32,
                 bottleneck: int = 16,
                 kernel_size: int = 3,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
                 dropout: float = 0.1,
                 seq_len: int = 50):
        super().__init__()
        self.seq_len = seq_len

        # Encoder
        enc_channels_list = [in_channels] + [enc_channels] * len(dilations)
        enc_blocks = []
        for i, d in enumerate(dilations):
            enc_blocks.append(TCNBlock(enc_channels_list[i], enc_channels_list[i + 1],
                                       kernel_size, d, dropout))
        self.encoder = nn.Sequential(*enc_blocks)
        self.enc_proj = nn.Linear(enc_channels, bottleneck)

        # Decoder
        self.dec_proj = nn.Linear(bottleneck, enc_channels)
        dec_channels_list = [enc_channels] * (len(dilations) + 1)
        dec_blocks = []
        for i, d in enumerate(dilations):
            dec_blocks.append(TCNBlock(dec_channels_list[i], dec_channels_list[i + 1],
                                       kernel_size, d, dropout))
        self.decoder = nn.Sequential(*dec_blocks)
        self.output_proj = nn.Linear(enc_channels, in_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1) → (B, 1, L)
        x_t = x.permute(0, 2, 1)
        enc = self.encoder(x_t)                      # (B, C, L)
        pooled = enc.mean(dim=2)                     # (B, C)
        return self.enc_proj(pooled)                 # (B, bottleneck)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_proj(z)                         # (B, C)
        h = h.unsqueeze(2).expand(-1, -1, self.seq_len)  # (B, C, L)
        dec = self.decoder(h)                        # (B, C, L)
        out = self.output_proj(dec.permute(0, 2, 1))  # (B, L, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


# ─────────────────────────────────────── dataset helper ────────────────────────

def make_windows(series: np.ndarray, window: int = 50,
                 stride: int = 25) -> np.ndarray:
    n = len(series)
    windows = []
    for start in range(0, n - window + 1, stride):
        windows.append(series[start:start + window])
    return np.array(windows, dtype=np.float32)[:, :, np.newaxis]  # (N, W, 1)


def make_labels(windows_start: np.ndarray, anomaly_idx: np.ndarray,
                window: int = 50) -> np.ndarray:
    """Label a window as anomalous if any ground-truth anomaly falls inside."""
    labels = np.zeros(len(windows_start), dtype=np.float32)
    anom_set = set(anomaly_idx.tolist())
    for i, start in enumerate(windows_start):
        for t in range(start, start + window):
            if t in anom_set:
                labels[i] = 1.0
                break
    return labels


# ─────────────────────────────────────── high-level wrappers ───────────────────

class TCNAnomalyClassifier:
    """Supervised TCN wrapper: trains on labeled windows."""

    def __init__(self, window: int = 50, stride: int = 25,
                 num_channels: int = 32, kernel_size: int = 3,
                 epochs: int = 20, lr: float = 1e-3,
                 batch_size: int = 128, device: str | None = None):
        self.window = window
        self.stride = stride
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TCNClassifier(1, num_channels, kernel_size).to(self.device)
        self.threshold = 0.5

    def fit(self, series: np.ndarray, anomaly_idx: np.ndarray,
            verbose: bool = True) -> "TCNAnomalyClassifier":
        windows = make_windows(series, self.window, self.stride)
        starts = np.arange(0, len(series) - self.window + 1, self.stride)
        labels = make_labels(starts, anomaly_idx, self.window)

        X = torch.from_numpy(windows)
        y = torch.from_numpy(labels).unsqueeze(1)
        loader = DataLoader(TensorDataset(X, y), self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

        for ep in range(1, self.epochs + 1):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            if verbose and (ep == 1 or ep % 5 == 0):
                print(f"  [TCN-Cls] Epoch {ep:3d}/{self.epochs}  loss={total/len(windows):.4f}")
        return self

    def predict(self, series: np.ndarray) -> dict:
        windows = make_windows(series, self.window, self.stride)
        starts = np.arange(0, len(series) - self.window + 1, self.stride)
        X = torch.from_numpy(windows).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X).squeeze(1).cpu().numpy()
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()

        # Map window scores back to per-timestep
        scores = np.zeros(len(series))
        counts = np.zeros(len(series))
        for prob, start in zip(probs, starts):
            scores[start:start + self.window] += float(prob)
            counts[start:start + self.window] += 1
        counts = np.where(counts == 0, 1, counts)
        scores /= counts

        anomaly_indices = np.where(scores > self.threshold)[0].tolist()
        return {"anomaly_indices": anomaly_indices, "scores": scores,
                "threshold": self.threshold}


class TCNAnomalyDetector:
    """Unsupervised TCN autoencoder wrapper: train on normal, detect via recon error."""

    def __init__(self, window: int = 50, stride: int = 25,
                 enc_channels: int = 32, bottleneck: int = 16,
                 kernel_size: int = 3, epochs: int = 20,
                 lr: float = 1e-3, batch_size: int = 128,
                 device: str | None = None):
        self.window = window
        self.stride = stride
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TCNAutoencoder(1, enc_channels, bottleneck, kernel_size,
                                    seq_len=window).to(self.device)
        self.threshold = 0.0

    def fit(self, series: np.ndarray, verbose: bool = True) -> "TCNAnomalyDetector":
        windows = make_windows(series, self.window, self.stride)
        X = torch.from_numpy(windows)
        loader = DataLoader(TensorDataset(X), self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

        for ep in range(1, self.epochs + 1):
            total = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device)
                recon = self.model(xb)
                loss = F.mse_loss(recon, xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            if verbose and (ep == 1 or ep % 5 == 0):
                print(f"  [TCN-AE] Epoch {ep:3d}/{self.epochs}  loss={total/len(windows):.6f}")

        # Compute threshold on training windows
        errors = self._compute_errors(windows)
        self.threshold = float(errors.mean() + 3 * errors.std())
        return self

    def _compute_errors(self, windows: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = torch.from_numpy(windows)
        loader = DataLoader(TensorDataset(X), 512, shuffle=False)
        errors = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                recon = self.model(xb)
                err = ((recon - xb) ** 2).mean(dim=(1, 2)).cpu().numpy()
                errors.append(err)
        return np.concatenate(errors)

    def score(self, series: np.ndarray) -> np.ndarray:
        windows = make_windows(series, self.window, self.stride)
        starts = np.arange(0, len(series) - self.window + 1, self.stride)
        errors = self._compute_errors(windows)
        scores = np.zeros(len(series))
        counts = np.zeros(len(series))
        for err, start in zip(errors, starts):
            scores[start:start + self.window] += float(err)
            counts[start:start + self.window] += 1
        counts = np.where(counts == 0, 1, counts)
        return scores / counts

    def predict(self, series: np.ndarray) -> dict:
        scores = self.score(series)
        anomaly_indices = np.where(scores > self.threshold)[0].tolist()
        return {"anomaly_indices": anomaly_indices, "scores": scores,
                "threshold": self.threshold}


# ─────────────────────────────────────── main ──────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from generate_data import generate_ecg

    print("=" * 55)
    print("TCN Anomaly Detection — Quick 5-Epoch Test")
    print("=" * 55)
    data, gt_idx = generate_ecg()
    train_data = data[:7000]

    print("\n--- Mode 1: Classification (Supervised) ---")
    cls = TCNAnomalyClassifier(window=50, stride=25, num_channels=16,
                                epochs=5, batch_size=64)
    cls.fit(train_data, gt_idx, verbose=True)
    result_cls = cls.predict(data)
    print(f"  Detected : {len(result_cls['anomaly_indices'])}  |  Ground truth: {len(gt_idx)}")

    print("\n--- Mode 2: Reconstruction (Unsupervised) ---")
    ae = TCNAnomalyDetector(window=50, stride=25, enc_channels=16,
                             bottleneck=8, epochs=5, batch_size=64)
    ae.fit(train_data, verbose=True)
    result_ae = ae.predict(data)
    print(f"  Threshold  : {result_ae['threshold']:.6f}")
    print(f"  Detected   : {len(result_ae['anomaly_indices'])}  |  Ground truth: {len(gt_idx)}")
    print("\nTCN quick test complete.")
