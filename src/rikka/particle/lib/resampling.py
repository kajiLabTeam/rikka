"""粒子重みの有効サンプル数計算と系統リサンプリング。

役割:
    正規化済み粒子重みの健全性評価と、低分散な親インデックス抽出を行う。
依存元:
    NumPyの配列演算と乱数生成器を利用する。
利用先:
    粒子フィルタの主ループが観測後のESS判定と再標本化に使用する。
処理フロー:
    重みからESSを計算し、必要時には一様オフセットから累積重みを探索して親を返す。
"""

import numpy as np


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """系統リサンプリングでインデックス配列を返す。O(N)・分散最小。"""
    n = len(weights)
    positions = (np.arange(n) + rng.uniform(0, 1)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


def _effective_sample_size(weights: np.ndarray) -> float:
    """正規化済み重みから有効サンプルサイズを返す。"""
    squared_sum = float(np.sum(np.square(weights)))
    if squared_sum <= 0.0 or not np.isfinite(squared_sum):
        return 0.0
    return 1.0 / squared_sum
