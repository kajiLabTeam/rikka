"""歩単位の運動観測を区間単位の移動状態へ復号する。

役割:
    端末方位、身体方位候補、移動軸を分離した観測列から、前進・左横歩き・
    右横歩きの連続区間を semi-Markov 動的計画法で推定する。端末の旋回は
    排他的な移動状態にせず、独立した診断値として保持する。
依存元:
    ``models`` から ``StepMotionObservation`` を取得し、標準ライブラリの角度・
    数値計算だけを利用する。設定の既定値や軌跡生成処理には依存しない。
利用先:
    将来 ``trajectory`` が既存の横歩き cluster 平滑化と比較・併用し、
    ``prepare_pdr_steps`` を通じて particle filter と同じ区間推定を共有する。
処理フロー:
    各観測を前進・左右横歩きの対数スコアへ変換し、区間統計から横歩きの
    最短長、移動軸の集中度、符号整合性を検証する。動的計画法で最良の区間列を
    復元し、歩ごとの状態、端末旋回尤度、区間診断を返す。
"""

from math import cos, exp, hypot, isfinite, log, sin
from typing import NamedTuple

from .models import StepMotionObservation

_FORWARD = "forward"
_SIDESTEP_LEFT = "sidestep_left"
_SIDESTEP_RIGHT = "sidestep_right"
_MOTION_MODES = (_FORWARD, _SIDESTEP_LEFT, _SIDESTEP_RIGHT)

_FULL_DISPLACEMENT_M = 0.08
_TURNING_THRESHOLD_RAD = 0.6108652381980153  # 35度
_MIN_SIDE_DURATION = 2
_MIN_RELIABLE_SIDE_STEPS = 2
_MIN_CALIBRATION_RELIABILITY = 0.45
_MIN_SIDE_STRENGTH = 0.30
_MIN_AXIS_CONCENTRATION = 0.45
_MIN_SIGNED_LATERAL_CONSISTENCY = 0.45
_MIN_SIDE_CONFIDENCE = 0.20
_SIDE_TRANSITION_PENALTY = 0.35
_INITIAL_SIDE_PENALTY = 0.85
_EPSILON = 1e-12


class DecodedMotionSegment(NamedTuple):
    """復号した連続区間と採用根拠の診断値。``end_index`` は排他的。"""

    start_index: int
    end_index: int
    motion_mode: str
    confidence: float
    evidence_count: int
    axis_concentration: float
    signed_lateral_consistency: float
    contains_device_turn: bool


class MotionDecodeResult(NamedTuple):
    """歩ごとの移動状態、端末旋回診断、区間診断をまとめた復号結果。"""

    motion_modes: tuple[str, ...]
    turn_likelihoods: tuple[float, ...]
    device_turn_flags: tuple[bool, ...]
    segments: tuple[DecodedMotionSegment, ...]


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


def _side_segment_stats(
    prefixes: _Prefixes,
    start: int,
    end: int,
) -> _SegmentStats:
    """横歩き候補区間の信頼度、軸集中度、符号整合性を返す。"""
    duration = end - start
    evidence_count = int(_range_sum(prefixes.reliable_count, start, end))
    mean_side_strength = _range_sum(prefixes.side_strength, start, end) / duration
    mean_calibration = _range_sum(prefixes.calibration, start, end) / duration
    signed_lateral_sum = _range_sum(prefixes.weighted_lateral, start, end)
    abs_lateral_sum = _range_sum(prefixes.weighted_abs_lateral, start, end)
    signed_consistency = abs(signed_lateral_sum) / max(abs_lateral_sum, _EPSILON)
    axis_weight = _range_sum(prefixes.axis_weight, start, end)
    axis_concentration = hypot(
        _range_sum(prefixes.axis_sin2, start, end),
        _range_sum(prefixes.axis_cos2, start, end),
    ) / max(axis_weight, _EPSILON)
    confidence = (
        mean_side_strength * axis_concentration * signed_consistency * mean_calibration
    )
    return _SegmentStats(
        confidence=_clip01(confidence),
        evidence_count=evidence_count,
        axis_concentration=_clip01(axis_concentration),
        signed_lateral_consistency=_clip01(signed_consistency),
        signed_lateral_sum=signed_lateral_sum,
        mean_calibration=mean_calibration,
        mean_side_strength=mean_side_strength,
    )


def _valid_side_segment(stats: _SegmentStats, state: int) -> bool:
    """区間が指定方向の高信頼な横歩き条件を満たすか返す。"""
    direction_matches = (
        stats.signed_lateral_sum > 0.0 if state == 1 else stats.signed_lateral_sum < 0.0
    )
    return (
        direction_matches
        and stats.evidence_count >= _MIN_RELIABLE_SIDE_STEPS
        and stats.mean_calibration >= _MIN_CALIBRATION_RELIABILITY
        and stats.mean_side_strength >= _MIN_SIDE_STRENGTH
        and stats.axis_concentration >= _MIN_AXIS_CONCENTRATION
        and stats.signed_lateral_consistency >= _MIN_SIGNED_LATERAL_CONSISTENCY
        and stats.confidence >= _MIN_SIDE_CONFIDENCE
    )


def _build_turn_likelihoods(
    observations: tuple[StepMotionObservation, ...] | list[StepMotionObservation],
) -> tuple[float, ...]:
    """単歩と近傍3歩の端末yaw変化から旋回診断値を作る。"""
    yaw_values = [
        _finite_or_zero(observation.yaw_delta) for observation in observations
    ]
    likelihoods: list[float] = []
    for index, yaw_delta in enumerate(yaw_values):
        start = max(0, index - 1)
        end = min(len(yaw_values), index + 2)
        neighborhood_yaw = abs(sum(yaw_values[start:end]))
        likelihoods.append(
            _clip01(max(abs(yaw_delta), neighborhood_yaw) / _TURNING_THRESHOLD_RAD)
        )
    return tuple(likelihoods)


def _transition_score(
    previous_state: int,
    state: int,
    boundary_turn_likelihood: float,
) -> float | None:
    """状態境界のスコアを返し、左右の直接遷移は許可しない。"""
    if previous_state == state:
        return None
    if previous_state != 0 and state != 0:
        return None
    turn_discount = 1.0 - 0.35 * boundary_turn_likelihood
    return -_SIDE_TRANSITION_PENALTY * turn_discount


def _decode_state_indexes(
    prefixes: _Prefixes,
    turn_likelihoods: tuple[float, ...],
    length: int,
) -> tuple[int, ...]:
    """semi-Markov動的計画法で最良の状態番号列を復元する。"""
    negative_infinity = float("-inf")
    scores = [
        [negative_infinity for _state in _MOTION_MODES] for _end in range(length + 1)
    ]
    back_pointers: list[list[tuple[int, int | None] | None]] = [
        [None for _state in _MOTION_MODES] for _end in range(length + 1)
    ]

    for end in range(1, length + 1):
        for candidate_state in range(len(_MOTION_MODES)):
            minimum_duration = 1 if candidate_state == 0 else _MIN_SIDE_DURATION
            for start in range(0, end - minimum_duration + 1):
                if candidate_state != 0:
                    segment_stats = _side_segment_stats(prefixes, start, end)
                    if not _valid_side_segment(segment_stats, candidate_state):
                        continue
                segment_score = _range_sum(
                    prefixes.emissions[candidate_state], start, end
                )
                if start == 0:
                    candidate = segment_score
                    if candidate_state != 0:
                        candidate -= _INITIAL_SIDE_PENALTY
                    if candidate > scores[end][candidate_state]:
                        scores[end][candidate_state] = candidate
                        back_pointers[end][candidate_state] = (start, None)
                    continue

                boundary_turn = max(
                    turn_likelihoods[start - 1],
                    turn_likelihoods[start],
                )
                for previous_state in range(len(_MOTION_MODES)):
                    transition = _transition_score(
                        previous_state,
                        candidate_state,
                        boundary_turn,
                    )
                    if (
                        transition is None
                        or scores[start][previous_state] == negative_infinity
                    ):
                        continue
                    candidate = (
                        scores[start][previous_state] + transition + segment_score
                    )
                    if candidate > scores[end][candidate_state]:
                        scores[end][candidate_state] = candidate
                        back_pointers[end][candidate_state] = (start, previous_state)

    final_state = max(
        range(len(_MOTION_MODES)),
        key=lambda state: scores[length][state],
    )
    if scores[length][final_state] == negative_infinity:
        return tuple(0 for _index in range(length))

    states = [0 for _index in range(length)]
    end = length
    trace_state: int | None = final_state
    while end > 0 and trace_state is not None:
        pointer = back_pointers[end][trace_state]
        if pointer is None:
            return tuple(0 for _index in range(length))
        start, trace_previous_state = pointer
        states[start:end] = [trace_state] * (end - start)
        end = start
        trace_state = trace_previous_state
    return tuple(states)


def _forward_segment_confidence(
    prefixes: _Prefixes,
    start: int,
    end: int,
) -> float:
    """前進区間の横歩き代替状態に対する平均スコア差を信頼度へ変換する。"""
    forward_score = _range_sum(prefixes.emissions[0], start, end)
    side_score = max(
        _range_sum(prefixes.emissions[1], start, end),
        _range_sum(prefixes.emissions[2], start, end),
    )
    mean_margin = (forward_score - side_score) / (end - start)
    return _clip01(1.0 / (1.0 + exp(-mean_margin)))


def _build_segments(
    state_indexes: tuple[int, ...],
    prefixes: _Prefixes,
    turn_flags: tuple[bool, ...],
) -> tuple[DecodedMotionSegment, ...]:
    """歩ごとの状態番号を連続区間へまとめ、診断値を付与する。"""
    if not state_indexes:
        return ()
    segments: list[DecodedMotionSegment] = []
    start = 0
    for end in range(1, len(state_indexes) + 1):
        if end < len(state_indexes) and state_indexes[end] == state_indexes[start]:
            continue
        state = state_indexes[start]
        stats = _side_segment_stats(prefixes, start, end)
        confidence = (
            _forward_segment_confidence(prefixes, start, end)
            if state == 0
            else stats.confidence
        )
        segments.append(
            DecodedMotionSegment(
                start_index=start,
                end_index=end,
                motion_mode=_MOTION_MODES[state],
                confidence=confidence,
                evidence_count=stats.evidence_count,
                axis_concentration=stats.axis_concentration,
                signed_lateral_consistency=stats.signed_lateral_consistency,
                contains_device_turn=any(turn_flags[start:end]),
            )
        )
        start = end
    return tuple(segments)


def decode_step_motion_segments(
    observations: tuple[StepMotionObservation, ...] | list[StepMotionObservation],
) -> MotionDecodeResult:
    """運動観測列を区間復号し、歩状態と端末旋回診断を返す。"""
    if not observations:
        return MotionDecodeResult((), (), (), ())
    features = [_observation_feature(observation) for observation in observations]
    prefixes = _build_prefixes(features)
    turn_likelihoods = _build_turn_likelihoods(observations)
    turn_flags = tuple(likelihood >= 1.0 for likelihood in turn_likelihoods)
    state_indexes = _decode_state_indexes(
        prefixes,
        turn_likelihoods,
        len(observations),
    )
    modes = tuple(_MOTION_MODES[state] for state in state_indexes)
    segments = _build_segments(state_indexes, prefixes, turn_flags)
    return MotionDecodeResult(modes, turn_likelihoods, turn_flags, segments)


def decode_step_motion_modes(
    observations: tuple[StepMotionObservation, ...] | list[StepMotionObservation],
) -> tuple[str, ...]:
    """運動観測列から歩ごとの前進・左右横歩き状態だけを返す。"""
    return decode_step_motion_segments(observations).motion_modes
