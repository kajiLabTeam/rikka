"""従来の ``rikka.analyze.pdr`` を維持する薄い互換 shim。

このモジュールは実装を持たず、公開 API と既存利用中の最小限の private helper を
新しい ``common`` / ``pdr`` / ``plot`` / ``cli`` 領域から再輸出する。
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
from ...common.lib.pdr_math import (
    _normalize_angle,
    _validate_non_negative_parameter,
    _validate_positive_parameter,
    _validate_scale,
)
from ...common.lib.sensors import load_sensor_data, process_sensor_data
from ...common.lib.time_utils import _sample_gyro_angle
from ...pdr.lib.fusion.adaptive import AdaptivePdrEstimator, estimate_adaptive_pdr
from ...pdr.lib.heading.body import (
    DynamicBodyHeadingEstimate,
    estimate_dynamic_body_headings,
)
from ...pdr.lib.heading.device_orientation import (
    _apply_device_orientation_to_horizontal,
    _classify_movement_type,
    _estimate_device_orientation_mode,
)
from ...pdr.lib.heading.motion import (
    _estimate_motion_heading_correction,
    _resolve_motion_heading_correction,
)
from ...pdr.lib.heading.resolver import (
    resolve_step_heading,
)
from ...pdr.lib.motion_state.clustering import _smooth_step_headings
from ...pdr.lib.motion_state.decoder import (
    DecodedMotionSegment,
    MotionDecodeResult,
    decode_step_motion_modes,
    decode_step_motion_segments,
)
from ...pdr.lib.motion_state.heading_policy import (
    _limit_heading_change,
    _stabilize_trajectory_headings,
)
from ...pdr.lib.motion_state.refinement import refine_step_headings_with_motion_model
from ...pdr.lib.motion_state.step_motion import (
    _is_sidestep_suspect_movement,
    _is_trajectory_sidestep_movement,
    estimate_step_motion,
)
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
from ...plot.lib.outputs import _build_trajectory_dataframe, _create_output_dir
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
    "_apply_device_orientation_to_horizontal",
    "_build_trajectory_dataframe",
    "_classify_movement_type",
    "_create_output_dir",
    "_estimate_device_orientation_mode",
    "_estimate_motion_heading_correction",
    "_is_sidestep_suspect_movement",
    "_is_trajectory_sidestep_movement",
    "_limit_heading_change",
    "_normalize_angle",
    "_resolve_motion_heading_correction",
    "_sample_gyro_angle",
    "_smooth_step_headings",
    "_stabilize_trajectory_headings",
    "_validate_non_negative_parameter",
    "_validate_positive_parameter",
    "_validate_scale",
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
