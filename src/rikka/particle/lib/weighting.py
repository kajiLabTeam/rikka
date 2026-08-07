"""地図・歩幅・運動状態尤度による粒子重み更新。

役割:
    提案粒子へ地図有効性と観測尤度を掛け、正規化前の事後重みを作る。
依存元:
    NumPy 配列だけを使用する。
利用先:
    particle runner の観測更新段階から呼び出す。
処理フロー:
    事前重みへ3種類の尤度を既存順で乗算して返す。
"""

import numpy as np


def weight(
    prior: np.ndarray,
    valid_transition: np.ndarray,
    stride_likelihood: np.ndarray,
    state_likelihood: np.ndarray,
    state_power: float,
) -> np.ndarray:
    """正規化前の事後重みを返す。"""
    return np.asarray(
        prior
        * valid_transition.astype(float)
        * stride_likelihood
        * np.power(state_likelihood, state_power),
        dtype=float,
    )
