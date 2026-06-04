# Time Series Anomaly Detection

Three anomaly detection methods on synthetic time series data. Pure NumPy + PyTorch.

## Structure

```
time-series-anomaly/
├── generate_data.py          # synthetic dataset generator (ECG, industrial, financial)
├── models/
│   ├── autoencoder.py        # LSTM Autoencoder (PyTorch)
│   ├── isolation_forest.py   # pure NumPy Isolation Forest
│   └── prophet_simple.py     # FFT-based trend+seasonality decomposition
├── detect.py                 # unified detection pipeline + visualization
└── evaluate.py               # Precision/Recall/F1/AUC benchmark
```

## Datasets

| Dataset     | Type                      | Anomalies               |
|-------------|---------------------------|-------------------------|
| ECG-like    | Periodic sinusoidal       | Spikes, missing beats   |
| Industrial  | Trend + seasonality       | Spikes, level shifts    |
| Financial   | Random walk               | Jump moves (>3σ)        |

Each dataset: 10,000 points, ~2% anomaly rate.

## Quick start

```bash
# Generate data
python generate_data.py

# Run detection (single method)
python detect.py

# Full benchmark (all methods × all datasets)
python evaluate.py
```

## Methods

- **LSTM Autoencoder**: window reconstruction error > mean+3σ threshold
- **Isolation Forest**: path-length anomaly score, windowed statistical features
- **ProphetSimple**: trend (moving average) + seasonal (FFT) decomposition; anomaly = |residual| > k×IQR
