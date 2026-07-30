"""運動状態 decoder の観測特徴量。

役割:
    観測値の正規化、prefix sum、区間統計型を実装する。
依存元:
    common の共有型と NumPy を利用する。
利用先:
    motion_state.decoder の区間スコア計算から使用される。
処理フロー:
    観測を有限な特徴量へ変換し、任意区間を定数時間で集計できる形にする。
"""

from math import cos, isfinite, log, sin
from typing import NamedTuple

from ....common.lib.models import StepMotionObservation

_FULL_DISPLACEMENT_M = 0.08
_MIN_SIDE_STRENGTH = 0.30
_EPSILON = 1e-12


class _ObservationFeature(NamedTuple):
    """動的計画法と区間診断で使用する1歩の数値特徴。"""

    emissions: tuple[float, float, float]
    quality: float
    side_strength: float
    calibration: float
    weighted_lateral: float
    weighted_abs_lateral: float
    axis_weight: float
    axis_sin2: float
    axis_cos2: float


class _SegmentStats(NamedTuple):
    """指定区間を横歩きとして評価した統計。"""

    confidence: float
    evidence_count: int
    axis_concentration: float
    signed_lateral_consistency: float
    signed_lateral_sum: float
    mean_calibration: float
    mean_side_strength: float


class _Prefixes(NamedTuple):
    """区間統計を定数時間で計算する累積和。"""

    emissions: tuple[list[float], list[float], list[float]]
    reliable_count: list[float]
    side_strength: list[float]
    calibration: list[float]
    weighted_lateral: list[float]
    weighted_abs_lateral: list[float]
    axis_weight: list[float]
    axis_sin2: list[float]
    axis_cos2: list[float]


def _clip01(value: float) -> float:
    """有限値を0から1へ制限し、非有限値を0として扱う。"""
    if not isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _finite_or_zero(value: float | None) -> float:
    """有限値をfloatで返し、欠損値と非有限値を0として扱う。"""
    if value is None or not isfinite(value):
        return 0.0
    return float(value)


def _safe_log_probability(value: float) -> float:
    """0を避けて確率相当値の対数を返す。"""
    return log(max(_EPSILON, min(1.0, value)))


def _observation_feature(observation: StepMotionObservation) -> _ObservationFeature:
    """区間復号用のスコアと集約可能な特徴を1歩の観測から作る。"""
    forward = _finite_or_zero(observation.forward_displacement)
    lateral = _finite_or_zero(observation.lateral_displacement)
    displacement = _finite_or_zero(observation.displacement_norm)
    calibration = _clip01(observation.calibration_reliability)
    quality = (
        _clip01(observation.motion_confidence)
        * calibration
        * _clip01(displacement / _FULL_DISPLACEMENT_M)
    )

    total_axis_displacement = abs(forward) + abs(lateral)
    lateral_share = abs(lateral) / max(total_axis_displacement, _EPSILON)

    has_axis = (
        observation.motion_axis_heading is not None
        and isfinite(observation.motion_axis_heading)
        and observation.body_heading_candidate is not None
        and isfinite(observation.body_heading_candidate)
    )
    if has_axis:
        assert observation.motion_axis_heading is not None
        assert observation.body_heading_candidate is not None
        relative_axis = (
            observation.motion_axis_heading - observation.body_heading_candidate
        )
        axis_cos2 = cos(2.0 * relative_axis)
        axis_sin2 = sin(2.0 * relative_axis)
        side_alignment = (1.0 - axis_cos2) / 2.0
        axis_weight = quality
    else:
        axis_cos2 = 0.0
        axis_sin2 = 0.0
        side_alignment = 0.5
        axis_weight = 0.0

    side_strength = quality * (0.7 * side_alignment + 0.3 * lateral_share)

    # 不確実な観測では前進へ戻す事前分布を持たせる。横歩きは高品質かつ
    # 区間として整合したときだけ動的計画法でこの差を逆転できる。
    forward_probability = 0.05 + 0.90 * (1.0 - side_strength)
    side_probability = 0.05 + 0.90 * side_strength
    emissions = (
        _safe_log_probability(forward_probability),
        _safe_log_probability(side_probability),
        _safe_log_probability(side_probability),
    )
    return _ObservationFeature(
        emissions=emissions,
        quality=quality,
        side_strength=side_strength,
        calibration=calibration,
        weighted_lateral=quality * lateral,
        weighted_abs_lateral=quality * abs(lateral),
        axis_weight=axis_weight,
        axis_sin2=axis_weight * axis_sin2,
        axis_cos2=axis_weight * axis_cos2,
    )


def _prefix(values: list[float]) -> list[float]:
    """区間和を求めるための先頭0付き累積和を返す。"""
    result = [0.0]
    total = 0.0
    for value in values:
        total += value
        result.append(total)
    return result


def _build_prefixes(features: list[_ObservationFeature]) -> _Prefixes:
    """すべての状態スコアと区間診断特徴の累積和を作る。"""
    return _Prefixes(
        emissions=(
            _prefix([feature.emissions[0] for feature in features]),
            _prefix([feature.emissions[1] for feature in features]),
            _prefix([feature.emissions[2] for feature in features]),
        ),
        reliable_count=_prefix(
            [
                float(
                    feature.quality >= _MIN_SIDE_STRENGTH
                    and feature.side_strength >= _MIN_SIDE_STRENGTH
                )
                for feature in features
            ]
        ),
        side_strength=_prefix([feature.side_strength for feature in features]),
        calibration=_prefix([feature.calibration for feature in features]),
        weighted_lateral=_prefix([feature.weighted_lateral for feature in features]),
        weighted_abs_lateral=_prefix(
            [feature.weighted_abs_lateral for feature in features]
        ),
        axis_weight=_prefix([feature.axis_weight for feature in features]),
        axis_sin2=_prefix([feature.axis_sin2 for feature in features]),
        axis_cos2=_prefix([feature.axis_cos2 for feature in features]),
    )


def _range_sum(prefix: list[float], start: int, end: int) -> float:
    """累積和から半開区間の合計を返す。"""
    return prefix[end] - prefix[start]
