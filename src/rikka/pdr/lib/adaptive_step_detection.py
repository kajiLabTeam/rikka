"""歩行周期に応じて近接ピークを除外する実験的なオフライン歩検出。

役割:
    固定0.5秒の間隔制限を避け、実時刻で推定した周期に余裕を持たせて歩を選ぶ。
依存元:
    common の設定・共有型・時刻検証、NumPy と SciPy のピーク検出を利用する。
利用先:
    step_detection.detect_step_result が adaptive_peak 指定時に呼び、通常PDRと
    PFへ同一の歩列と候補診断を渡す。
処理フロー:
    高さを満たす候補を抽出し、前後3秒の信号を等間隔化して自己相関を求める。
    周期・活動量・突出度で候補を評価し、強い候補から実時刻で近接重複を除く。
    前後の信号と候補を使うためリアルタイム・因果的な検出器ではない。
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ...common.config import PEAK_HEIGHT, SAMPLING_RATE
from ...common.lib.models import StepCandidateDiagnostic, StepDetectionResult
from ...common.lib.time_utils import _time_values

# 対応域と安全弁は実験的な固定値。既存peak方式の設定には影響しない。
WINDOW_HALF_S = 3.0
PERIOD_MIN_S = 0.33
PERIOD_MAX_S = 1.2
MIN_CORRELATION = 0.45
MIN_ACTIVITY_STD = 0.06
INTERVAL_RATIO = 0.65
FALLBACK_INTERVAL_S = 0.4


def _local_period(
    times: np.ndarray,
    signal: np.ndarray,
    center: float,
    activity_signal: np.ndarray,
) -> tuple[float | None, float, float, float]:
    """局所周期・相関・活動量・振幅範囲を返す。歩の採用結果には依存しない。"""
    start, end = np.searchsorted(
        times, [center - WINDOW_HALF_S, center + WINDOW_HALF_S]
    )
    local_t = times[start:end]
    local_y = signal[start:end]
    if len(local_t) < 3:
        return None, 0.0, 0.0, 0.0
    activity = float(np.std(activity_signal[start:end]))
    local_peaks, props = find_peaks(local_y, height=PEAK_HEIGHT, prominence=0.0)
    strengths = props["prominences"]
    # 信号の包絡の上下幅ではなく、局所ピークの突出度を強度の基準にする。
    span = float(np.percentile(strengths, 90)) if len(strengths) else 0.0
    dt = float(np.median(np.diff(local_t)))
    if (
        local_t[-1] - local_t[0] < 2 * PERIOD_MAX_S
        or np.max(np.diff(local_t)) > max(0.1, 5 * dt)
        or activity < MIN_ACTIVITY_STD
    ):
        return None, 0.0, activity, span
    # 不規則な時刻を均等化するが、欠測の長い窓は上で周期不明として扱う。
    rate = min(100.0, 1.0 / dt)
    grid = np.arange(local_t[0], local_t[-1], 1.0 / rate)
    values = np.interp(grid, local_t, local_y)
    # 歩行開始・停止や姿勢変化に伴う低周波の包絡を周期と誤認しない。
    trend = (
        pd.Series(values)
        .rolling(max(3, int(round(1.5 * rate))), center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    values -= trend
    first = max(1, int(np.ceil(PERIOD_MIN_S * rate)))
    last = min(len(values) // 2, int(np.floor(PERIOD_MAX_S * rate)))
    if last <= first + 1:
        return None, 0.0, activity, span
    correlations = []
    for lag in range(first, last + 1):
        left, right = values[:-lag], values[lag:]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlations.append(
            float(np.dot(left, right) / denominator) if denominator > 0 else 0.0
        )
    corr = np.clip(np.asarray(correlations), -1.0, 1.0)
    maxima, _ = find_peaks(corr)
    best = float(np.max(corr))
    # 複数周期の倍数を選びにくくするため、最大相関に近い最短の極大を使う。
    plausible = maxima[corr[maxima] >= max(MIN_CORRELATION, 0.85 * best)]
    period = None
    confidence = max(0.0, best)
    if len(plausible):
        chosen = int(plausible[0])
        period, confidence = (first + chosen) / rate, float(corr[chosen])
    # 左右の強度差や周期の揺れで自己相関が2歩周期を選ぶ場合を独立な候補列で点検。
    # 採用済み歩列は参照せず、強い候補のうち0.25秒以内の重複だけを除く。
    strong: list[int] = []
    if len(strengths):
        cutoff = max(0.02, 0.25 * float(np.percentile(strengths, 90)))
        for i in np.argsort(-strengths):
            if strengths[i] >= cutoff and all(
                abs(local_t[local_peaks[i]] - local_t[j]) >= 0.25 for j in strong
            ):
                strong.append(int(local_peaks[i]))
    gaps = np.diff(local_t[sorted(strong)])
    # 対応域より短い副峰間隔と、長い停止・欠落区間は周期の中央値へ混ぜない。
    gaps = gaps[(gaps >= PERIOD_MIN_S) & (gaps <= PERIOD_MAX_S)]
    if len(gaps) >= 3:
        median = float(np.median(gaps))
        support = float(np.mean(np.abs(gaps - median) <= 0.3 * median))
        if PERIOD_MIN_S <= median <= PERIOD_MAX_S and support >= 0.7:
            if period is None or 1.65 <= period / median <= 2.35:
                period, confidence = median, support
        if period is not None:
            shorter = [
                int(i)
                for i in maxima
                if 1.65 <= period / ((first + i) / rate) <= 2.35
                and corr[i] >= max(0.2, 0.3 * confidence)
            ]
            for i in sorted(shorter, key=lambda i: -corr[i]):
                alternative = (first + i) / rate
                support = float(
                    np.mean(np.abs(gaps - alternative) <= 0.3 * alternative)
                )
                if support >= 0.4:
                    period, confidence = alternative, float(corr[i])
                    break
    return period, confidence, activity, span


def detect_adaptive_peaks(df_acc: pd.DataFrame) -> StepDetectionResult:
    """実時刻と局所周期から歩を選び、全候補の診断を返す。"""
    signal = np.asarray(df_acc["low_lin_norm"], dtype=float)
    activity_signal = np.asarray(
        df_acc.get("lin_norm", df_acc["low_lin_norm"]), dtype=float
    )
    times = _time_values(df_acc)
    if times is None:
        times = np.arange(len(signal), dtype=float) / SAMPLING_RATE
    if not np.isfinite(signal).all() or not np.isfinite(activity_signal).all():
        raise ValueError("low_lin_norm 列には有限な数値だけを指定してください。")
    candidates, properties = find_peaks(signal, height=PEAK_HEIGHT, prominence=0.0)
    prominences = properties["prominences"]
    periods: list[float | None] = []
    confidences, intervals, reasons = [], [], []
    cache: dict[int, tuple[float | None, float, float, float]] = {}
    for index, prominence in zip(candidates, prominences, strict=True):
        # 0.5秒ごとの窓を共有し、候補数に比例した自己相関の再計算を避ける。
        key = int(round(float(times[index]) * 2))
        if key not in cache:
            cache[key] = _local_period(times, signal, key / 2, activity_signal)
        period, confidence, activity, span = cache[key]
        interval = (
            FALLBACK_INTERVAL_S
            if period is None
            else float(np.clip(INTERVAL_RATIO * period, 0.25, 0.65))
        )
        periods.append(period)
        confidences.append(confidence)
        intervals.append(interval)
        if activity < MIN_ACTIVITY_STD:
            reasons.append("low_activity")
        elif prominence < max(0.02, 0.15 * span):
            reasons.append("low_prominence")
        else:
            reasons.append("eligible")
    kept: list[int] = []
    # 候補強度順の非最大抑制。周期から欠落歩を生成する処理は行わない。
    priority = sorted(
        range(len(candidates)),
        key=lambda i: (-prominences[i], -signal[candidates[i]], candidates[i]),
    )
    for i in priority:
        if reasons[i] != "eligible":
            continue
        if any(
            abs(times[candidates[i]] - times[candidates[j]])
            < max(intervals[i], intervals[j])
            for j in kept
        ):
            reasons[i] = "close_to_stronger_peak"
        else:
            kept.append(i)
            reasons[i] = "accepted" if periods[i] is not None else "accepted_fallback"
    accepted = set(kept)
    diagnostics = tuple(
        StepCandidateDiagnostic(
            sample_index=int(index),
            timestamp_s=float(times[index]),
            period_s=periods[i],
            confidence=float(confidences[i]),
            min_interval_s=float(intervals[i]),
            prominence=float(prominences[i]),
            accepted=i in accepted,
            reason=reasons[i],
        )
        for i, index in enumerate(candidates)
    )
    return StepDetectionResult(
        method="adaptive_peak",
        peaks=np.asarray(sorted(candidates[i] for i in kept), dtype=int),
        segments=(),
        threshold=PEAK_HEIGHT,
        polarity=None,
        diagnostics=diagnostics,
    )
