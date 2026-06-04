"""
models/isolation_forest.py — Pure NumPy Isolation Forest.

Algorithm
---------
Build N isolation trees.  Each tree randomly partitions data by:
  1. Selecting a random feature.
  2. Selecting a random split point between min and max of that feature.
Anomaly score = average path length normalised by expected path length c(n).
Score in [0, 1]; score > 0.5 → anomaly.

Parameters
----------
n_estimators : 100 trees
max_samples  : 256  (subsampling per tree)
contamination: 0.1  (used to set decision threshold)
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────── tree node ─────────────────────────────

class _IsolationNode:
    __slots__ = ("left", "right", "feature", "threshold", "size", "depth")

    def __init__(self):
        self.left: "_IsolationNode | None" = None
        self.right: "_IsolationNode | None" = None
        self.feature: int = -1
        self.threshold: float = 0.0
        self.size: int = 0
        self.depth: int = 0


# ─────────────────────────────────────── helpers ───────────────────────────────

def _c(n: int) -> float:
    """Expected path length of an unsuccessful BST search through n samples."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def _build_tree(X: np.ndarray, depth: int, max_depth: int, rng: np.random.Generator
                ) -> _IsolationNode:
    node = _IsolationNode()
    node.size = len(X)
    node.depth = depth

    if depth >= max_depth or len(X) <= 1:
        return node

    # Pick a random feature and a random split within [min, max]
    feat = int(rng.integers(X.shape[1]))
    col = X[:, feat]
    lo, hi = col.min(), col.max()
    if lo == hi:
        return node   # no split possible

    thresh = float(rng.uniform(lo, hi))
    node.feature = feat
    node.threshold = thresh

    left_mask = col < thresh
    right_mask = ~left_mask
    node.left  = _build_tree(X[left_mask],  depth + 1, max_depth, rng)
    node.right = _build_tree(X[right_mask], depth + 1, max_depth, rng)
    return node


def _path_length(node: _IsolationNode, x: np.ndarray, current_depth: int) -> float:
    if node.left is None and node.right is None:
        return current_depth + _c(node.size)
    if x[node.feature] < node.threshold:
        if node.left is None:
            return current_depth + 1.0 + _c(node.size)
        return _path_length(node.left, x, current_depth + 1)
    else:
        if node.right is None:
            return current_depth + 1.0 + _c(node.size)
        return _path_length(node.right, x, current_depth + 1)


# ─────────────────────────────────────── main class ────────────────────────────

class IsolationForest:
    """
    Pure NumPy Isolation Forest.

    Usage
    -----
    >>> clf = IsolationForest(n_estimators=100, max_samples=256)
    >>> clf.fit(X_train)
    >>> scores = clf.score(X_test)  # higher = more anomalous
    >>> labels = clf.predict(X_test)  # -1 = anomaly, 1 = normal
    """

    def __init__(self, n_estimators: int = 100, max_samples: int = 256,
                 contamination: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self._trees: list[_IsolationNode] = []
        self._max_depth: int = 0
        self.threshold_: float = 0.0

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray) -> "IsolationForest":
        """
        X : shape (n_samples, n_features)
        """
        X = np.atleast_2d(X).astype(float)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        sub = min(self.max_samples, n)
        self._max_depth = int(np.ceil(np.log2(sub))) + 1

        self._trees = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=sub, replace=False)
            tree = _build_tree(X[idx], depth=0, max_depth=self._max_depth, rng=rng)
            self._trees.append(tree)

        # calibrate threshold using training scores
        train_scores = self.score(X)
        k = max(1, int(n * self.contamination))
        sorted_scores = np.sort(train_scores)[::-1]
        self.threshold_ = float(sorted_scores[k - 1])
        return self

    # ------------------------------------------------------------------ score
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores in [0, 1].
        Scores closer to 1 → more anomalous.
        """
        X = np.atleast_2d(X).astype(float)
        sub = self.max_samples
        cn = _c(sub)
        all_lengths = np.zeros((len(X), self.n_estimators))

        for j, tree in enumerate(self._trees):
            for i, x in enumerate(X):
                all_lengths[i, j] = _path_length(tree, x, 0)

        avg_lengths = all_lengths.mean(axis=1)
        # Anomaly score: 2^(-E[h(x)] / c(n))
        scores = 2.0 ** (-avg_lengths / (cn + 1e-12))
        return scores

    # ------------------------------------------------------------------ predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns array of 1 (normal) or -1 (anomaly)."""
        scores = self.score(X)
        labels = np.where(scores >= self.threshold_, -1, 1)
        return labels

    def predict_as_bool(self, X: np.ndarray) -> np.ndarray:
        """Returns boolean mask: True = anomaly."""
        return self.predict(X) == -1


# ─────────────────────────────────────── feature extraction ────────────────────

def extract_features(series: np.ndarray, window: int = 50,
                     step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract statistical features from sliding windows.
    Returns (features, center_indices).
    """
    n = len(series)
    features, centers = [], []
    for start in range(0, n - window + 1, step):
        seg = series[start:start + window]
        feats = [
            seg.mean(),
            seg.std(),
            seg.max() - seg.min(),
            float(np.percentile(seg, 75) - np.percentile(seg, 25)),  # IQR
            float(np.abs(np.diff(seg)).mean()),                       # mean abs diff
        ]
        features.append(feats)
        centers.append(start + window // 2)
    return np.array(features, dtype=float), np.array(centers, dtype=int)


class IsolationForestDetector:
    """
    Wraps IsolationForest for per-timestep anomaly detection via windowed features.
    """

    def __init__(self, window: int = 50, step: int = 5,
                 n_estimators: int = 100, max_samples: int = 256,
                 contamination: float = 0.02):
        self.window = window
        self.step = step
        self.clf = IsolationForest(n_estimators, max_samples, contamination)

    def fit(self, series: np.ndarray) -> "IsolationForestDetector":
        features, _ = extract_features(series, self.window, self.step)
        self.clf.fit(features)
        return self

    def predict(self, series: np.ndarray) -> dict:
        features, centers = extract_features(series, self.window, self.step)
        scores_win = self.clf.score(features)

        # Map back to per-timestep scores
        scores = np.zeros(len(series))
        counts = np.zeros(len(series))
        for score, center in zip(scores_win, centers):
            lo = max(0, center - self.window // 2)
            hi = min(len(series), center + self.window // 2)
            scores[lo:hi] += score
            counts[lo:hi] += 1
        counts = np.where(counts == 0, 1, counts)
        scores /= counts

        threshold = self.clf.threshold_
        anomaly_indices = np.where(scores > threshold)[0].tolist()
        return {
            "anomaly_indices": anomaly_indices,
            "scores": scores,
            "threshold": threshold,
        }


# ─────────────────────────────────────── main ──────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from generate_data import generate_ecg

    print("Testing Isolation Forest on ECG data …")
    data, gt_idx = generate_ecg()

    detector = IsolationForestDetector(window=50, step=10, contamination=0.02)
    detector.fit(data[:7000])
    result = detector.predict(data)
    print(f"Threshold          : {result['threshold']:.4f}")
    print(f"Detected anomalies : {len(result['anomaly_indices'])}")
    print(f"Ground truth       : {len(gt_idx)}")
