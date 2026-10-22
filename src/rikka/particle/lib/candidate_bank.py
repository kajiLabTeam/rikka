"""通常運動モデルから作った有限候補の選択と重み補正。

役割:
    親ごとの候補因子Gから採用候補を抽出し、平均因子S/Kを返す。
依存元:
    NumPyの配列演算と乱数生成器、標準のdataclassを利用する。
利用先:
    proposeが完全な候補状態を選択し、recorderが候補と採用状態を保存する。
処理フロー:
    非負有限の候補因子を検証し、正の質量を持つ親だけから候補を抽出する。
    全候補無効の親は未採用とし、既存の補充・集団復旧へ渡す。
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParticleCandidateBank:
    """入力親ごとの全候補。選択前状態を保持し、更新後状態と区別する。"""

    positions: np.ndarray
    headings: np.ndarray
    step_lengths: np.ndarray
    heading_drift: np.ndarray
    stride_scale: np.ndarray
    motion_state: np.ndarray
    factors: np.ndarray
    valid_transition: np.ndarray
    selected_indices: np.ndarray
    mean_factors: np.ndarray


def select_candidate_bank(
    factors: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Gに比例して候補を選び、選択番号と重み増分S/Kを返す。

    全候補の因子が0の親は選択番号を-1にする。K=1では乱数を消費せず、
    救済なしの単一提案と同じ乱数系列を維持する。
    """
    factors = np.asarray(factors, dtype=float)
    if factors.ndim != 2 or factors.shape[1] < 1:
        raise ValueError("候補因子は候補数1以上の2次元配列で指定してください")
    if not np.all(np.isfinite(factors)) or np.any(factors < 0.0):
        raise ValueError("候補因子は有限の非負値で指定してください")
    # 最大値で割ってから和を取ることで、有限値の和によるoverflowを避ける。
    maxima = np.max(factors, axis=1)
    live = maxima > 0.0
    scaled = np.zeros_like(factors)
    np.divide(factors, maxima[:, None], out=scaled, where=live[:, None])
    totals = scaled.sum(axis=1)
    means = maxima * (totals / factors.shape[1])
    selected = np.full(len(factors), -1, dtype=int)
    if factors.shape[1] == 1:
        selected[live] = 0
        return selected, means
    probabilities = scaled[live] / totals[live, None]
    cumulative = np.cumsum(probabilities, axis=1)
    if len(cumulative):
        cumulative[:, -1] = 1.0
        draws = rng.random(np.count_nonzero(live))
        selected[live] = np.sum(draws[:, None] >= cumulative, axis=1)
    return selected, means
