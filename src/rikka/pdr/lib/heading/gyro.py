"""PDR の歩区間境界決定。

役割:
    歩区間境界決定を独立した部品として実装する。
依存元:
    common の設定・共有型・時刻処理と同じ heading 領域の部品を利用する。
利用先:
    pdr pipeline または heading resolver から使用される。
処理フロー:
    ピーク列または接地区間から対象範囲を返す。
"""

import numpy as np

from ....common.config import (
    STEP_LENGTH_WINDOW,
)
from ....common.lib.models import StepSegment


def _step_segment_bounds(
    peaks: np.ndarray,
    i: int,
    n_samples: int,
    step_segments: tuple[StepSegment, ...] = (),
) -> tuple[int, int] | None:
    """加速度方位推定に使うステップ区間を返す。"""
    if i < len(step_segments):
        segment = step_segments[i]
        return segment.start_index, segment.end_index
    if i + 1 < len(peaks):
        return int(peaks[i]), int(peaks[i + 1])
    if i < len(peaks):
        peak = int(peaks[i])
        return (
            max(0, peak - STEP_LENGTH_WINDOW),
            min(n_samples, peak + STEP_LENGTH_WINDOW + 1),
        )
    return None


step_segment_bounds = _step_segment_bounds
