"""通常 PDR の推定方式レジストリ。

役割:
    legacy / adaptive / robust 推定を同じ選択表から取得できるようにする。
依存元:
    同じ ``fusion`` 領域の各実装を参照する。
利用先:
    PDR pipeline が設定された推定方式を分岐なしで選択するために使用する。
処理フロー:
    設定名をレジストリで解決し、選択された推定 callable を呼び出し元へ返す。
"""

from collections.abc import Callable
from dataclasses import dataclass

from ....common.lib.models import (
    StepDirectionPosterior,
    StepHeading,
    StepLengthObservation,
    StepMotionEvidence,
    StepMotionPosterior,
)
from ..motion_state.evidence import (
    build_step_motion_evidences,
    build_step_motion_observations,
)
from .adaptive import estimate_adaptive_pdr
from .robust import resolve_step_directions


@dataclass(frozen=True)
class MotionEstimationResult:
    """推定方式に依存しない歩方位・歩幅・診断の結果。"""

    step_headings: list[StepHeading]
    step_lengths: list[float]
    motion_evidences: tuple[StepMotionEvidence, ...]
    motion_posteriors: tuple[StepMotionPosterior, ...] = ()
    direction_posteriors: tuple[StepDirectionPosterior, ...] = ()


def _estimate_legacy(
    step_headings: list[StepHeading],
    step_lengths: list[float],
    _length_observations: tuple[StepLengthObservation, ...],
    motion_evidences: tuple[StepMotionEvidence, ...],
    _smoothing_mode: str,
    _direction_fixed_lag: int,
) -> MotionEstimationResult:
    """既存の補正済み結果を変更せず返す。"""
    return MotionEstimationResult(
        step_headings,
        step_lengths,
        motion_evidences,
    )


def _estimate_adaptive(
    step_headings: list[StepHeading],
    _step_lengths: list[float],
    length_observations: tuple[StepLengthObservation, ...],
    motion_evidences: tuple[StepMotionEvidence, ...],
    smoothing_mode: str,
    _direction_fixed_lag: int,
) -> MotionEstimationResult:
    """adaptive 推定結果を共通結果へ変換する。"""
    result = estimate_adaptive_pdr(
        step_headings,
        length_observations,
        motion_evidences,
        smoothing_mode,
    )
    return MotionEstimationResult(
        result.step_headings,
        result.step_lengths,
        motion_evidences,
        result.posteriors,
    )


def _estimate_robust(
    step_headings: list[StepHeading],
    step_lengths: list[float],
    _length_observations: tuple[StepLengthObservation, ...],
    _motion_evidences: tuple[StepMotionEvidence, ...],
    smoothing_mode: str,
    direction_fixed_lag: int,
) -> MotionEstimationResult:
    """robust 推定結果を共通結果へ変換する。"""
    observations = build_step_motion_observations(step_headings)
    resolved, posteriors = resolve_step_directions(
        step_headings,
        observations,
        smoothing_mode=smoothing_mode,
        fixed_lag=direction_fixed_lag,
    )
    return MotionEstimationResult(
        resolved,
        step_lengths,
        build_step_motion_evidences(resolved),
        direction_posteriors=posteriors,
    )


MotionEstimatorFunction = Callable[
    [
        list[StepHeading],
        list[float],
        tuple[StepLengthObservation, ...],
        tuple[StepMotionEvidence, ...],
        str,
        int,
    ],
    MotionEstimationResult,
]

MOTION_ESTIMATORS: dict[str, MotionEstimatorFunction] = {
    "legacy": _estimate_legacy,
    "adaptive": _estimate_adaptive,
    "robust": _estimate_robust,
}
