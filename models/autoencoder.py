"""
models/autoencoder.py — LSTM Autoencoder for time-series anomaly detection.

Architecture
------------
  Encoder : LSTM(input=1, hidden=32, layers=2) → linear → latent 16-dim
  Decoder : repeat latent → LSTM(hidden=16→32, layers=2) → Linear → 1

Window size : 50 timesteps
Anomaly     : reconstruction error > mean_train + 3 × std_train
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────── dataset ───────────────────────────────

def make_windows(series: np.ndarray, window: int = 50) -> np.ndarray:
    """Slide a window over series → (N_windows, window, 1)."""
    n = len(series)
    windows = np.lib.stride_tricks.sliding_window_view(series, window)
    return windows[:, :, np.newaxis].astype(np.float32)


# ─────────────────────────────────────── model ─────────────────────────────────

class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32,
                 num_layers: int = 2, latent_dim: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, 1)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden) — use last layer
        latent = self.fc(h_n[-1])          # (batch, latent_dim)
        return latent


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_size: int = 32,
                 num_layers: int = 2, output_size: int = 1, seq_len: int = 50):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc_in = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim)
        h0 = self.fc_in(z)                         # (batch, hidden)
        # repeat as input sequence
        inp = h0.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq, hidden)
        out, _ = self.lstm(inp)                     # (batch, seq, hidden)
        recon = self.fc_out(out)                    # (batch, seq, 1)
        return recon


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int = 50, hidden_size: int = 32,
                 latent_dim: int = 16, num_layers: int = 2):
        super().__init__()
        self.encoder = LSTMEncoder(1, hidden_size, num_layers, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, num_layers, 1, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


# ─────────────────────────────────────── trainer ───────────────────────────────

class AnomalyAutoencoder:
    """
    High-level wrapper: fit on (clean) train data, score any series.
    """

    def __init__(self, window: int = 50, hidden: int = 32, latent: int = 16,
                 num_layers: int = 2, lr: float = 1e-3, epochs: int = 20,
                 batch_size: int = 256, device: str | None = None):
        self.window = window
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = (device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LSTMAutoencoder(window, hidden, latent, num_layers).to(self.device)
        self.threshold: float = 0.0
        self._train_errors: np.ndarray | None = None

    # ---------------------------------------------------------------- fit
    def fit(self, series: np.ndarray, verbose: bool = True) -> "AnomalyAutoencoder":
        windows = make_windows(series, self.window)   # (N, W, 1)
        X = torch.from_numpy(windows)
        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.model.train()

        for ep in range(1, self.epochs + 1):
            total_loss = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device)
                recon = self.model(xb)
                loss = criterion(recon, xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(xb)
            avg = total_loss / len(X)
            if verbose and (ep % 5 == 0 or ep == 1):
                print(f"  Epoch {ep:3d}/{self.epochs}  loss={avg:.6f}")

        # compute threshold on training data
        train_errors = self._compute_errors(windows)
        self._train_errors = train_errors
        self.threshold = float(train_errors.mean() + 3 * train_errors.std())
        return self

    # ---------------------------------------------------------------- score
    def _compute_errors(self, windows: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = torch.from_numpy(windows)
        loader = DataLoader(TensorDataset(X), batch_size=512, shuffle=False)
        errors = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                recon = self.model(xb)
                err = ((recon - xb) ** 2).mean(dim=(1, 2)).cpu().numpy()
                errors.append(err)
        return np.concatenate(errors)

    def score(self, series: np.ndarray) -> np.ndarray:
        """Return per-timestep anomaly score (reconstruction error)."""
        windows = make_windows(series, self.window)
        errors = self._compute_errors(windows)
        # Map window error back to original time axis (use last step of window)
        scores = np.zeros(len(series))
        for i, e in enumerate(errors):
            scores[i + self.window - 1] = e
        return scores

    def predict(self, series: np.ndarray) -> dict:
        scores = self.score(series)
        anomaly_mask = scores > self.threshold
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        return {
            "anomaly_indices": anomaly_indices,
            "scores": scores,
            "threshold": self.threshold,
        }


# ─────────────────────────────────────── main ──────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from generate_data import generate_ecg

    print("Testing LSTM Autoencoder on ECG data …")
    data, gt_idx = generate_ecg()
    # Use first 7000 as train (assumed mostly normal), rest for test
    train, test = data[:7000], data

    ae = AnomalyAutoencoder(epochs=10, batch_size=128)
    ae.fit(train, verbose=True)
    result = ae.predict(test)
    print(f"Threshold        : {result['threshold']:.6f}")
    print(f"Detected anomalies: {len(result['anomaly_indices'])}")
    print(f"Ground truth      : {len(gt_idx)}")
