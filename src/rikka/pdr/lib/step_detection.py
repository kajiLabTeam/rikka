"""PDR のステップ検出手法。

役割:
    線形加速度ノルムのピーク方式、または上下加速度の閾値方式で歩行イベントを検出し、
    ピーク位置・1歩区間・診断値を ``StepDetectionResult`` にまとめる。
依存元:
    ``config`` から距離・高さ・区間長・閾値、``models`` から結果型を取得し、
    NumPy、Pandas、SciPy のピーク検出を利用する。
利用先:
    ``trajectory.prepare_pdr_steps`` と ``sensor_plot`` がステップ列を作るために使い、
    互換 facade から従来の ``detect_steps`` も公開される。
処理フロー:
    指定方式を検証し、候補イベントを抽出して近接重複や不正区間を除き、結果を返す。
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ...common.config import (
    MAX_SEG_SAMPLES,
    MIN_SEG_SAMPLES,
    PEAK_DISTANCE,
    PEAK_HEIGHT,
    STEP_DETECTION_METHOD,
    STEP_VERTICAL_SMOOTH_WINDOW,
    STEP_VERTICAL_THRESHOLD_PERCENTILE,
)
from ...common.lib.models import StepDetectionResult, StepSegment
from ...common.lib.validation import (
    STEP_DETECTION_METHODS as STEP_DETECTION_METHODS,
)
from ...common.lib.validation import validate_choice


def _validate_step_detection_method(method: str) -> str:
    """ステップ検出手法名を検証する。"""
    return validate_choice(
        "step_detection_method",
        method,
        STEP_DETECTION_METHODS,
    )


def _detect_steps_by_peak(df_acc: pd.DataFrame) -> StepDetectionResult:
    """既存方式: 平滑化線形加速度ノルムからステップピークを検出する。"""
    peaks, _ = find_peaks(
        df_acc["low_lin_norm"].to_numpy(),
        distance=PEAK_DISTANCE,
        height=PEAK_HEIGHT,
    )
    return StepDetectionResult(
        method="peak",
        peaks=np.asarray(peaks),
        segments=(),
        threshold=PEAK_HEIGHT,
        polarity=None,
    )


def _threshold_groups(mask: np.ndarray) -> list[tuple[int, int]]:
    """True が連続する範囲を [start, end) のリストで返す。"""
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            groups.append((start, i))
            start = None
    if start is not None:
        groups.append((start, len(mask)))
    return groups


def _suppress_close_contacts(
    contacts: list[tuple[int, float]],
    min_distance: int = PEAK_DISTANCE,
) -> list[tuple[int, float]]:
    """近すぎる接地候補は強度が大きい方だけ残す。"""
    if not contacts:
        return []

    kept: list[tuple[int, float]] = [contacts[0]]
    for index, strength in contacts[1:]:
        prev_index, prev_strength = kept[-1]
        if index - prev_index < min_distance:
            if strength > prev_strength:
                kept[-1] = (index, strength)
        else:
            kept.append((index, strength))
    return kept


def _detect_steps_by_vertical_threshold(df_acc: pd.DataFrame) -> StepDetectionResult:
    """論文方式に寄せて、上下加速度の接地閾値から1歩区間を抽出する。"""
    values = np.asarray(pd.to_numeric(df_acc["v_acc"], errors="coerce"), dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return StepDetectionResult(
            method="paper_vertical_threshold",
            peaks=np.array([], dtype=int),
            segments=(),
            threshold=None,
            polarity=None,
        )

    filled = values.copy()
    median = float(np.nanmedian(filled[finite]))
    filled[~finite] = median
    smoothed = (
        pd.Series(filled)
        .rolling(window=STEP_VERTICAL_SMOOTH_WINDOW, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )

    positive_span = float(np.nanpercentile(smoothed, 95))
    negative_span = abs(float(np.nanpercentile(smoothed, 5)))
    polarity = -1 if negative_span > positive_span else 1
    raw_contact_signal = filled * polarity
    contact_signal = smoothed * polarity
    finite_contact = contact_signal[np.isfinite(contact_signal)]
    threshold = float(
        np.nanpercentile(finite_contact, STEP_VERTICAL_THRESHOLD_PERCENTILE)
    )
    baseline = float(np.nanmedian(finite_contact))
    signal_max = float(np.nanmax(finite_contact))
    if threshold <= baseline:
        threshold = baseline + (signal_max - baseline) * 0.25
    if signal_max <= baseline:
        return StepDetectionResult(
            method="paper_vertical_threshold",
            peaks=np.array([], dtype=int),
            segments=(),
            threshold=threshold,
            polarity=polarity,
        )

    groups = _threshold_groups(contact_signal >= threshold)
    contacts: list[tuple[int, float]] = []
    for start, end in groups:
        if end <= start:
            continue
        segment = raw_contact_signal[start:end]
        if not np.isfinite(segment).any():
            continue
        local_index = int(np.nanargmax(segment))
        contact_index = start + local_index
        contacts.append((contact_index, float(raw_contact_signal[contact_index])))

    contacts = _suppress_close_contacts(contacts)
    contact_indexes = [index for index, _strength in contacts]

    segments: list[StepSegment] = []
    for start_index, end_index in zip(
        contact_indexes[:-1], contact_indexes[1:], strict=False
    ):
        seg_len = end_index - start_index
        if MIN_SEG_SAMPLES <= seg_len <= MAX_SEG_SAMPLES:
            segments.append(
                StepSegment(
                    start_index=int(start_index),
                    end_index=int(end_index),
                    contact_index=int(end_index),
                )
            )

    peaks = np.asarray([segment.contact_index for segment in segments], dtype=int)
    return StepDetectionResult(
        method="paper_vertical_threshold",
        peaks=peaks,
        segments=tuple(segments),
        threshold=threshold,
        polarity=polarity,
    )


def detect_step_result(
    df_acc: pd.DataFrame,
    method: str | None = None,
) -> StepDetectionResult:
    """指定方式でステップを検出し、互換ピーク列と区間情報を返す。"""
    selected_method = _validate_step_detection_method(
        STEP_DETECTION_METHOD if method is None else method
    )
    if selected_method == "paper_vertical_threshold":
        return _detect_steps_by_vertical_threshold(df_acc)
    return _detect_steps_by_peak(df_acc)


def detect_steps(df_acc: pd.DataFrame, method: str | None = None) -> np.ndarray:
    """指定方式でステップを検出し、既存互換のピーク配列だけを返す。"""
    return detect_step_result(df_acc, method).peaks
