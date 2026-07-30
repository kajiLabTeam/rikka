"""PDR の加速度ピーク方位推定。

役割:
    加速度ピーク方位推定を独立した部品として実装する。
依存元:
    common の設定・共有型・時刻処理と同じ heading 領域の部品を利用する。
利用先:
    pdr pipeline または heading resolver から使用される。
処理フロー:
    加速度ピークを選び、2方式の方位候補を返す。
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ....common.config import (
    ACCEL_HEADING_MIN_LINE_LENGTH,
    ACCEL_HEADING_MIN_PEAK_DISTANCE,
    ACCEL_HEADING_MIN_PEAK_NORM,
    MAX_SEG_SAMPLES,
    MIN_SEG_SAMPLES,
)
from ....common.lib.models import StepSegment
from ....common.lib.pdr_math import (
    _normalize_angle,
    _score_ratio,
)
from .gyro import _step_segment_bounds


class _AccelHeadingResult(NamedTuple):
    """加速度方位候補の内部計算結果。"""

    method1_heading: float | None
    method2_heading: float | None
    confidence: float
    segment_start_index: int | None
    segment_end_index: int | None
    peak1_index: int | None
    peak2_index: int | None


def _select_two_accel_peaks(norm: np.ndarray) -> tuple[int, int] | None:
    """平面加速度ノルムから方位推定用の2つの極大点を選ぶ。"""
    if len(norm) < 2 or not np.isfinite(norm).any():
        return None

    safe_norm = np.where(np.isfinite(norm), norm, -np.inf)
    peak_indexes, _ = find_peaks(
        safe_norm,
        distance=max(1, ACCEL_HEADING_MIN_PEAK_DISTANCE),
    )
    candidates = list(peak_indexes)

    # 端点にピークが出るデータでは find_peaks が拾えないため、強い点を補助候補にする。
    for index in np.argsort(safe_norm)[::-1]:
        int_index = int(index)
        if safe_norm[int_index] == -np.inf:
            continue
        if int_index not in candidates:
            candidates.append(int_index)
        if len(candidates) >= 4:
            break

    selected: list[int] = []
    for index in sorted(candidates, key=lambda idx: safe_norm[idx], reverse=True):
        if all(index != existing for existing in selected):
            selected.append(int(index))
        if len(selected) == 2:
            break

    if len(selected) < 2:
        return None
    peak_a, peak_b = sorted(selected[:2])
    return peak_a, peak_b


def _estimate_accel_headings(
    df_acc: pd.DataFrame,
    peaks: np.ndarray,
    i: int,
    direction_offset: float,
    step_segments: tuple[StepSegment, ...] = (),
) -> _AccelHeadingResult:
    """論文手法1/2の加速度平面成分方位と信頼度を返す。"""
    bounds = _step_segment_bounds(peaks, i, len(df_acc), step_segments)
    if bounds is None:
        return _AccelHeadingResult(None, None, 0.0, None, None, None, None)

    start, end = bounds
    if end <= start:
        return _AccelHeadingResult(None, None, 0.0, start, end, None, None)

    h_y = df_acc["h_y"].iloc[start:end].to_numpy(dtype=float)
    h_z = df_acc["h_z"].iloc[start:end].to_numpy(dtype=float)
    valid = np.isfinite(h_y) & np.isfinite(h_z)
    if valid.sum() < 3:
        return _AccelHeadingResult(None, None, 0.0, start, end, None, None)

    h_y_safe = np.where(valid, h_y, np.nan)
    h_z_safe = np.where(valid, h_z, np.nan)
    norm = np.sqrt(h_y_safe**2 + h_z_safe**2)
    selected = _select_two_accel_peaks(norm)
    if selected is None:
        return _AccelHeadingResult(None, None, 0.0, start, end, None, None)

    local_peak1, local_peak2 = selected
    peak1_index = start + local_peak1
    peak2_index = start + local_peak2
    point1 = np.array([h_y_safe[local_peak1], h_z_safe[local_peak1]], dtype=float)
    point2 = np.array([h_y_safe[local_peak2], h_z_safe[local_peak2]], dtype=float)
    if not np.isfinite(point1).all() or not np.isfinite(point2).all():
        return _AccelHeadingResult(
            None, None, 0.0, start, end, peak1_index, peak2_index
        )

    norm1 = float(norm[local_peak1])
    norm2 = float(norm[local_peak2])
    line_length = float(np.linalg.norm(point2 - point1))
    peak_distance = abs(local_peak2 - local_peak1)
    seg_len = end - start

    if line_length <= 1e-12:
        return _AccelHeadingResult(
            None, None, 0.0, start, end, peak1_index, peak2_index
        )

    # 手法1: 時間的に早い極大値方向。手法2: ノルムが大きい極大値方向。
    method1_vec = point1 - point2
    method2_vec = point1 - point2 if norm1 >= norm2 else point2 - point1
    method1_heading = _normalize_angle(
        float(np.arctan2(method1_vec[1], method1_vec[0])) + direction_offset
    )
    method2_heading = _normalize_angle(
        float(np.arctan2(method2_vec[1], method2_vec[0])) + direction_offset
    )

    strength_score = _score_ratio(min(norm1, norm2), ACCEL_HEADING_MIN_PEAK_NORM)
    separation_score = _score_ratio(peak_distance, ACCEL_HEADING_MIN_PEAK_DISTANCE)
    line_length_score = _score_ratio(line_length, ACCEL_HEADING_MIN_LINE_LENGTH)
    duration_score = 1.0 if MIN_SEG_SAMPLES <= seg_len <= MAX_SEG_SAMPLES else 0.0
    confidence = strength_score * separation_score * line_length_score * duration_score

    return _AccelHeadingResult(
        method1_heading=method1_heading,
        method2_heading=method2_heading,
        confidence=float(confidence),
        segment_start_index=start,
        segment_end_index=end,
        peak1_index=peak1_index,
        peak2_index=peak2_index,
    )


estimate_accel_headings = _estimate_accel_headings
select_two_accel_peaks = _select_two_accel_peaks
