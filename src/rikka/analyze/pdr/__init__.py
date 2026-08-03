"""従来の ``rikka.analyze.pdr`` を維持する薄い互換 shim。

このモジュールは実装を持たず、公開 API を新しい ``common`` / ``pdr`` /
``plot`` / ``cli`` 領域から再輸出する。private helper は互換対象に含めない。
"""
# ruff: noqa: F401

from ...cli.commands import run
from ...common.config import SIDESTEP_LENGTH_SCALE, TURNING_LENGTH_SCALE
from ...common.lib.gyro_bias import estimate_gyro_bias
from ...common.lib.models import (
    AdaptivePdrResult,
    AdaptivePdrState,
    GyroBiasResult,
    PreparedPdrSteps,
    StepDetectionResult,
    StepDirectionPosterior,
    StepHeading,
    StepLengthObservation,
    StepMotion,
    StepMotionEvidence,
    StepMotionObservation,
    StepMotionPosterior,
    StepSegment,
)
from ...common.lib.sensors import load_sensor_data, process_sensor_data
from ...pdr.lib.fusion.adaptive import AdaptivePdrEstimator, estimate_adaptive_pdr
from ...pdr.lib.heading.body import (
    DynamicBodyHeadingEstimate,
    estimate_dynamic_body_headings,
)
from ...pdr.lib.heading.resolver import (
    resolve_step_heading,
)
from ...pdr.lib.motion_state.decoder import (
    DecodedMotionSegment,
    MotionDecodeResult,
    decode_step_motion_modes,
    decode_step_motion_segments,
)
from ...pdr.lib.motion_state.refinement import refine_step_headings_with_motion_model
from ...pdr.lib.motion_state.step_motion import estimate_step_motion
from ...pdr.lib.preparation import prepare_pdr_steps
from ...pdr.lib.step_detection import detect_step_result, detect_steps
from ...pdr.lib.step_length import (
    estimate_step_length,
    estimate_step_length_forward,
)
from ...pdr.lib.trajectory import (
    estimate_trajectory,
    estimate_trajectory_with_headings,
)
from ...plot.lib.trajectory import plot_trajectory

__all__ = [
    "AdaptivePdrEstimator",
    "AdaptivePdrResult",
    "AdaptivePdrState",
    "DecodedMotionSegment",
    "DynamicBodyHeadingEstimate",
    "GyroBiasResult",
    "MotionDecodeResult",
    "PreparedPdrSteps",
    "SIDESTEP_LENGTH_SCALE",
    "StepDetectionResult",
    "StepDirectionPosterior",
    "StepHeading",
    "StepLengthObservation",
    "StepMotion",
    "StepMotionEvidence",
    "StepMotionObservation",
    "StepMotionPosterior",
    "StepSegment",
    "TURNING_LENGTH_SCALE",
    "decode_step_motion_modes",
    "decode_step_motion_segments",
    "detect_step_result",
    "detect_steps",
    "estimate_adaptive_pdr",
    "estimate_dynamic_body_headings",
    "estimate_gyro_bias",
    "estimate_step_length",
    "estimate_step_length_forward",
    "estimate_step_motion",
    "estimate_trajectory",
    "estimate_trajectory_with_headings",
    "load_sensor_data",
    "plot_trajectory",
    "prepare_pdr_steps",
    "process_sensor_data",
    "refine_step_headings_with_motion_model",
    "resolve_step_heading",
    "run",
]
