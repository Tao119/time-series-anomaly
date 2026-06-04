"""
models/lstm_vae.py — LSTM Variational Autoencoder for Unsupervised Anomaly Detection

Architecture
------------
  Encoder: LSTM(hidden=64) → two linear heads → μ, log σ²
  Reparameterization trick: z = μ + ε × σ
  Decoder: LSTM reconstructs the sequence from z
  Loss: ELBO = Reconstruction Loss + β × KL divergence  (β=0.1)
  Anomaly score: reconstruction loss (ELBO reconstruction term)

Training: on normal data only.
Inference: high reconstruction error → anomaly.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────── model ─────────────────────────────────

class LSTMEncoder(nn.Module):
    """LSTM encoder → μ and log σ²."""

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 1, latent_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc_mu     = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq, input_size)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]                                  # (batch, hidden)
        mu     = self.fc_mu(h)                       # (batch, latent)
        logvar = self.fc_logvar(h)                   # (batch, latent)
        return mu, logvar


class LSTMDecoder(nn.Module):
    """LSTM decoder: latent z → reconstructed sequence."""

    def __init__(self, latent_dim: int = 32, hidden_size: int = 64,
                 num_layers: int = 1, output_size: int = 1,
                 seq_len: int = 50):
        super().__init__()
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.fc_in = nn.Linear(latent_dim, hidden_size)
        self.lstm  = nn.LSTM(hidden_size, hidden_size, num_layers,
                              batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent)
        h0 = self.fc_in(z)                           # (batch, hidden)
        inp = h0.unsqueeze(1).repeat(1, self.seq_len, 1)   # (batch, seq, hidden)
        out, _ = self.lstm(inp)                      # (batch, seq, hidden)
        recon = self.fc_out(out)                     # (batch, seq, output)
        return recon


class LSTMVAE(nn.Module):
    """
    LSTM Variational Autoencoder.

    Parameters
    ----------
    input_size  : feature dimension per timestep (1 for univariate)
    hidden_size : LSTM hidden size
    latent_dim  : VAE bottleneck dimension
    seq_len     : window length
    beta        : KL weight (β-VAE)
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 1, latent_dim: int = 32,
                 seq_len: int = 50, beta: float = 0.1):
        super().__init__()
        self.beta = beta
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, num_layers,
                                   input_size, seq_len)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """z = μ + ε × σ,  ε ~ N(0, I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # Deterministic at inference

    def forward(self, x: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def elbo_loss(self, x: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ELBO = E[log p(x|z)] - β × KL(q(z|x) || p(z))
        Returns (total_loss, recon_loss, kl_loss) — all scalars.
        """
        recon, mu, logvar = self(x)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        # KL: -0.5 × Σ(1 + log σ² - μ² - σ²)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample reconstruction error (anomaly score).
        Returns shape (batch,).
        """
        self.eval()
        with torch.no_grad():
            recon, _, _ = self(x)
            err = ((recon - x) ** 2).mean(dim=(1, 2))   # (batch,)
        return err


# ─────────────────────────────────────── dataset helper ────────────────────────

def make_windows(series: np.ndarray, window: int = 50,
                 stride: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (windows, start_indices).
    windows: (N, window, 1) float32
    """
    n = len(series)
    windows, starts = [], []
    for s in range(0, n - window + 1, stride):
        windows.append(series[s:s + window])
        starts.append(s)
    arr = np.array(windows, dtype=np.float32)[:, :, np.newaxis]
    return arr, np.array(starts, dtype=np.int64)


# ─────────────────────────────────────── high-level wrapper ────────────────────

class LSTMVAEDetector:
    """
    Train LSTM-VAE on normal data; detect anomalies via ELBO reconstruction term.
    """

    def __init__(self, window: int = 50, stride: int = 25,
                 hidden_size: int = 64, latent_dim: int = 32,
                 num_layers: int = 1, beta: float = 0.1,
                 epochs: int = 20, lr: float = 1e-3,
                 batch_size: int = 256, device: str | None = None):
        self.window      = window
        self.stride      = stride
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMVAE(1, hidden_size, num_layers, latent_dim,
                              window, beta).to(self.device)
        self.threshold   = 0.0
        self._train_errors: np.ndarray | None = None

    def fit(self, series: np.ndarray, verbose: bool = True) -> "LSTMVAEDetector":
        windows, _ = make_windows(series, self.window, self.stride)
        X = torch.from_numpy(windows)
        loader = DataLoader(TensorDataset(X), self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for ep in range(1, self.epochs + 1):
            self.model.train()
            total_loss = total_recon = total_kl = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device)
                loss, recon_l, kl_l = self.model.elbo_loss(xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss  += loss.item()  * len(xb)
                total_recon += recon_l.item() * len(xb)
                total_kl    += kl_l.item()  * len(xb)
            n = len(windows)
            if verbose and (ep == 1 or ep % 5 == 0):
                print(f"  [LSTM-VAE] Epoch {ep:3d}/{self.epochs}"
                      f"  loss={total_loss/n:.6f}"
                      f"  recon={total_recon/n:.6f}"
                      f"  kl={total_kl/n:.6f}")

        # Threshold calibration on training windows
        train_errors = self._window_errors(windows)
        self._train_errors = train_errors
        self.threshold = float(train_errors.mean() + 3 * train_errors.std())
        return self

    def _window_errors(self, windows: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = torch.from_numpy(windows)
        loader = DataLoader(TensorDataset(X), 512, shuffle=False)
        errors = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                err = self.model.reconstruction_error(xb)
                errors.append(err.cpu().numpy())
        return np.concatenate(errors)

    def score(self, series: np.ndarray) -> np.ndarray:
        """Per-timestep anomaly score."""
        windows, starts = make_windows(series, self.window, self.stride)
        errors = self._window_errors(windows)
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
        return {
            "anomaly_indices": anomaly_indices,
            "scores": scores,
            "threshold": self.threshold,
        }


# ─────────────────────────────────────── main ──────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from generate_data import generate_ecg

    print("=" * 55)
    print("LSTM-VAE Anomaly Detection — Quick 5-Epoch Test")
    print("=" * 55)
    data, gt_idx = generate_ecg()
    train_data = data[:7000]

    detector = LSTMVAEDetector(
        window=50, stride=25,
        hidden_size=32, latent_dim=16,
        epochs=5, batch_size=128,
        beta=0.1,
    )
    detector.fit(train_data, verbose=True)
    result = detector.predict(data)

    print(f"\nThreshold         : {result['threshold']:.6f}")
    print(f"Detected anomalies: {len(result['anomaly_indices'])}")
    print(f"Ground truth      : {len(gt_idx)}")

    # Quick metrics
    detected = set(result["anomaly_indices"])
    gt_set   = set(gt_idx.tolist())
    tp = len(detected & gt_set)
    fp = len(detected - gt_set)
    fn = len(gt_set - detected)
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print("\nLSTM-VAE quick test complete.")
