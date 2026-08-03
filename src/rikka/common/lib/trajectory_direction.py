"""軌跡終端の進行方向を比較する共通指標。

役割:
    点密度や軌跡長が異なる候補を正規化弧長へ揃え、終端区間の方向差、基準方向
    への投影、逆向き接線の割合を計算する。
依存元:
    NumPy から補間、ベクトル、角度の配列演算を取得する。
利用先:
    エージェント評価と互換shimが正解軌跡に対する終端逆走を検出し、
    複数計測の代表軌跡選択が
    正解を使わずに計測間の終端方向整合性を評価するために使用する。
処理フロー:
    始点を合わせて弧長補間し、終端15%の変位ベクトルと対応接線を比較して、方向差、
    cosine、逆向き割合、および複合的な失敗判定を返す。
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerminalDirectionMetrics:
    """候補軌跡の終端方向を基準軌跡と比較した結果。"""

    direction_error_deg: float
    progress_cosine: float
    opposed_fraction: float
    failure: bool


def sample_trajectory_by_arclength(
    points: np.ndarray,
    count: int = 300,
) -> np.ndarray:
    """軌跡を始点基準の正規化弧長上へ補間する。"""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points は shape=(N, 2) を指定してください")
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("points は有限な2点以上の軌跡を指定してください")
    if count < 3:
        raise ValueError("count は3以上を指定してください")

    centered = values - values[0]
    keep = np.r_[True, np.linalg.norm(np.diff(centered, axis=0), axis=1) > 1e-9]
    filtered = centered[keep]
    if len(filtered) < 2:
        raise ValueError("軌跡長が0です")
    distances = np.r_[
        0.0,
        np.cumsum(np.linalg.norm(np.diff(filtered, axis=0), axis=1)),
    ]
    targets = np.linspace(0.0, distances[-1], count)
    return np.column_stack(
        [np.interp(targets, distances, filtered[:, axis]) for axis in range(2)]
    )


def terminal_heading(sampled: np.ndarray, fraction: float = 0.15) -> float:
    """正規化済み軌跡の終端区間を結ぶ方位をラジアンで返す。"""
    values = np.asarray(sampled, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
        raise ValueError("sampled は shape=(N, 2) の3点以上を指定してください")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction は0より大きく1より小さくしてください")
    start = max(
        0,
        min(
            len(values) - 2,
            int(np.floor((1.0 - fraction) * (len(values) - 1))),
        ),
    )
    displacement = values[-1] - values[start]
    if float(np.linalg.norm(displacement)) <= 1e-9:
        raise ValueError("終端区間の変位が0です")
    return float(np.arctan2(displacement[1], displacement[0]))


def evaluate_terminal_direction(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    fraction: float = 0.15,
) -> TerminalDirectionMetrics:
    """候補と基準の終端方向を比較し、逆走判定を返す。"""
    candidate_values = np.asarray(candidate, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    if candidate_values.shape != reference_values.shape:
        raise ValueError("candidate と reference のshapeを一致させてください")
    candidate_angle = terminal_heading(candidate_values, fraction)
    reference_angle = terminal_heading(reference_values, fraction)
    angle_difference = float(
        np.arctan2(
            np.sin(candidate_angle - reference_angle),
            np.cos(candidate_angle - reference_angle),
        )
    )
    progress_cosine = float(np.cos(angle_difference))

    start = max(
        0,
        min(
            len(candidate_values) - 2,
            int(np.floor((1.0 - fraction) * (len(candidate_values) - 1))),
        ),
    )
    candidate_steps = np.diff(candidate_values[start:], axis=0)
    reference_steps = np.diff(reference_values[start:], axis=0)
    candidate_norms = np.linalg.norm(candidate_steps, axis=1)
    reference_norms = np.linalg.norm(reference_steps, axis=1)
    valid = (candidate_norms > 1e-9) & (reference_norms > 1e-9)
    if not valid.any():
        raise ValueError("終端区間に比較可能な接線がありません")
    cosines = np.sum(candidate_steps[valid] * reference_steps[valid], axis=1) / (
        candidate_norms[valid] * reference_norms[valid]
    )
    opposed_fraction = float(np.mean(cosines < 0.0))
    return TerminalDirectionMetrics(
        direction_error_deg=float(np.degrees(abs(angle_difference))),
        progress_cosine=progress_cosine,
        opposed_fraction=opposed_fraction,
        failure=progress_cosine < 0.0 or opposed_fraction >= 0.5,
    )


def _circular_mean(angles: np.ndarray) -> float:
    """角度列の円平均を返す。"""
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def _circular_distance(left: float, right: float) -> float:
    """2方位間の絶対wrapped角度差を返す。"""
    return abs(float(np.arctan2(np.sin(left - right), np.cos(left - right))))


def terminal_consensus_outliers(
    sampled: np.ndarray,
    data_names: list[str],
    families: list[str],
    *,
    threshold_deg: float = 90.0,
) -> tuple[set[int], float, np.ndarray]:
    """計測・方式を等重みにした終端方向と逆向き候補を返す。"""
    values = np.asarray(sampled, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("sampled は shape=(C, N, 2) を指定してください")
    if len(values) != len(data_names) or len(values) != len(families):
        raise ValueError("候補数とdata_names/familiesの長さを一致させてください")
    if len(set(data_names)) < 3:
        raise ValueError("終端方向コンセンサスには3計測以上が必要です")
    if not 0.0 < threshold_deg <= 180.0:
        raise ValueError("threshold_deg は0より大きく180以下にしてください")

    headings = np.asarray([terminal_heading(item) for item in values], dtype=float)
    data_headings: list[float] = []
    for data_name in sorted(set(data_names)):
        family_headings: list[float] = []
        for family in sorted(
            {
                family_name
                for candidate_data, family_name in zip(
                    data_names, families, strict=True
                )
                if candidate_data == data_name
            }
        ):
            indices = [
                index
                for index, (candidate_data, candidate_family) in enumerate(
                    zip(data_names, families, strict=True)
                )
                if candidate_data == data_name and candidate_family == family
            ]
            family_headings.append(_circular_mean(headings[indices]))
        data_headings.append(_circular_mean(np.asarray(family_headings)))

    data_values = np.asarray(data_headings, dtype=float)
    costs = np.asarray(
        [
            sum(_circular_distance(candidate, other) for other in data_values)
            for candidate in data_values
        ]
    )
    consensus = float(data_values[int(np.argmin(costs))])
    errors = np.asarray(
        [np.degrees(_circular_distance(heading, consensus)) for heading in headings],
        dtype=float,
    )
    return set(np.flatnonzero(errors > threshold_deg).tolist()), consensus, errors


__all__ = [
    "TerminalDirectionMetrics",
    "evaluate_terminal_direction",
    "sample_trajectory_by_arclength",
    "terminal_consensus_outliers",
    "terminal_heading",
]
