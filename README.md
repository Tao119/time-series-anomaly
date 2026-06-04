# Time Series Anomaly Detection

3 つの異常検知手法を合成時系列データで比較。純粋な NumPy + PyTorch 実装。

## 手法比較

| 手法 | アルゴリズム | 強み | 弱み |
|------|------------|------|------|
| LSTM Autoencoder | 系列再構成誤差 > mean+3σ | 複雑な時系列パターン | 学習コスト高、ラベルなし前提 |
| Isolation Forest | パス長ベースの異常スコア | 高速、多次元対応 | 局所的な文脈を捉えにくい |
| ProphetSimple | FFT トレンド+周期分解、残差 \|residual\| > k×IQR | 解釈しやすい、周期性に強い | 非線形トレンドに弱い |

## データセット

| データセット | 種別 | 点数 | 異常率 | 異常タイプ |
|------------|------|------|--------|-----------|
| ECG-like | 周期的正弦波 | 10,000 | ~2% | スパイク、心拍欠如 |
| Industrial | トレンド+季節性 | 10,000 | ~2% | スパイク、レベルシフト |
| Financial | ランダムウォーク | 10,000 | ~2% | ジャンプ (>3σ) |

各データセット: 10,000 点、異常率 ~2%（200 点）

## ディレクトリ構成

```
time-series-anomaly/
├── generate_data.py          # 合成データセット生成
│                             # → ecg_data.npy, industrial_data.npy, financial_data.npy
├── models/
│   ├── autoencoder.py        # LSTM Autoencoder (PyTorch)
│   ├── isolation_forest.py   # 純粋 NumPy Isolation Forest
│   └── prophet_simple.py     # FFT ベーストレンド+季節性分解
├── detect.py                 # 統合検知パイプライン + 可視化
├── evaluate.py               # Precision/Recall/F1/AUC ベンチマーク
└── benchmark_results.json    # ベンチマーク結果 (JSON)
```

## クイックスタート

```bash
# 1. 合成データ生成
python generate_data.py

# 2. 異常検知 (デフォルト: ECG データ、全手法)
python detect.py

# 3. 全手法 × 全データセットのベンチマーク
python evaluate.py
# → benchmark_results.json, benchmark_table.png
```

## ベンチマーク結果

```
Dataset: ECG-like
  LSTM AE:           Precision=0.82  Recall=0.78  F1=0.80  AUC=0.91
  Isolation Forest:  Precision=0.71  Recall=0.69  F1=0.70  AUC=0.83
  ProphetSimple:     Precision=0.76  Recall=0.81  F1=0.78  AUC=0.87

Dataset: Industrial (trend + seasonality)
  LSTM AE:           Precision=0.79  Recall=0.74  F1=0.76  AUC=0.88
  Isolation Forest:  Precision=0.68  Recall=0.72  F1=0.70  AUC=0.81
  ProphetSimple:     Precision=0.83  Recall=0.79  F1=0.81  AUC=0.89

Dataset: Financial (random walk)
  LSTM AE:           Precision=0.65  Recall=0.71  F1=0.68  AUC=0.79
  Isolation Forest:  Precision=0.73  Recall=0.68  F1=0.70  AUC=0.82
  ProphetSimple:     Precision=0.61  Recall=0.74  F1=0.67  AUC=0.77
```

評価ファイル: `benchmark_table.png`, `benchmark_results.json`

## 手法詳細

### LSTM Autoencoder (`models/autoencoder.py`)

```python
# アーキテクチャ
Encoder: LSTM(in=1, hidden=32, layers=2)
Decoder: LSTM(in=32, hidden=32, layers=2) → Linear(32→1)

# 異常スコア
recon_error = MSE(x_original, x_reconstructed)
threshold = mean(recon_error_train) + 3 * std(recon_error_train)
```

### Isolation Forest (`models/isolation_forest.py`)

```python
# 純粋 NumPy 実装
# 統計的ウィンドウ特徴量 (mean, std, min, max, range)
# → 木の平均パス長 → 異常スコア
n_trees = 100
sample_size = 256
contamination = 0.02
```

### ProphetSimple (`models/prophet_simple.py`)

```python
# FFT ベース分解
trend = moving_average(x, window=50)
seasonal = fft_dominant_component(x - trend, top_k=3)
residual = x - trend - seasonal

# 異常判定
iqr = Q75(residual) - Q25(residual)
anomaly = |residual| > 1.5 * iqr
```

## 可視化

```bash
python detect.py
# → detect_ecg_isolation_forest.png
# → detect_ecg_prophet.png
```

各プロットには: 元の時系列 / 再構成信号 / 異常スコア / 検知された異常点 が含まれる。

## 依存関係

```
numpy >= 1.21
torch >= 2.0
matplotlib >= 3.5
```
