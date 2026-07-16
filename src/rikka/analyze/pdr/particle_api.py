"""Particle filter から利用する PDR 内部 API の公開 bridge。

役割:
    particle filter が必要とする PDR 内部関数だけを安定した公開名へ付け替え、
    通常 PDR と確率的 PDR の依存境界を1か所にまとめる。
依存元:
    ``common``、``heading``、``models``、``plotting``、``sidestep``、
    ``step_length``、``time_utils`` から共有する検証・推定・描画 API を取得する。
利用先:
    ``analyze.particle`` 配下の内部実装だけがこの bridge を import し、PDR の
    private helper や互換 facade への直接依存を避ける。
処理フロー:
    独自計算は行わず、必要な関数・型を意味の明確な公開名として再 export する。
"""

from .common import (
    _validate_forward_heading_source as validate_forward_heading_source,
)
from .common import (
    _validate_motion_heading_correction as validate_motion_heading_correction,
)
from .common import (
    _validate_non_negative_parameter as validate_non_negative_parameter,
)
from .common import (
    _validate_positive_parameter as validate_positive_parameter,
)
from .common import (
    _validate_sidestep_heading_source as validate_sidestep_heading_source,
)
from .common import (
    _validate_sidestep_smoothing as validate_sidestep_smoothing,
)
from .common import (
    _validate_sidestep_suspect_mode as validate_sidestep_suspect_mode,
)
from .heading import (
    _estimate_device_orientation_mode as estimate_device_orientation_mode,
)
from .heading import (
    _resolve_motion_heading_correction as resolve_motion_heading_correction,
)
from .heading import (
    resolve_step_heading,
)
from .models import (
    StepHeading,
    StepMotionEvidence,
    StepMotionObservation,
    StepMotionPosterior,
    StepSegment,
)
from .plotting import (
    _compute_pixel_coords as compute_pixel_coords,
)
from .plotting import (
    _plot_heading_overlay as plot_heading_overlay,
)
from .sidestep import (
    _smooth_step_headings as smooth_step_headings,
)
from .sidestep import (
    _stabilize_trajectory_headings as stabilize_trajectory_headings,
)
from .sidestep import (
    build_particle_motion_headings,
    build_step_motion_evidences,
    build_step_motion_observations,
    estimate_step_motion,
)
from .step_length import (
    _estimate_initial_forward_angle as estimate_initial_forward_angle,
)
from .step_length import (
    estimate_step_length,
    estimate_step_length_forward,
)
from .time_utils import _step_output_time as step_output_time

__all__ = [
    "StepHeading",
    "StepMotionEvidence",
    "StepMotionObservation",
    "StepMotionPosterior",
    "StepSegment",
    "build_step_motion_evidences",
    "build_step_motion_observations",
    "build_particle_motion_headings",
    "compute_pixel_coords",
    "estimate_device_orientation_mode",
    "estimate_initial_forward_angle",
    "estimate_step_length",
    "estimate_step_length_forward",
    "estimate_step_motion",
    "plot_heading_overlay",
    "resolve_motion_heading_correction",
    "resolve_step_heading",
    "smooth_step_headings",
    "stabilize_trajectory_headings",
    "step_output_time",
    "validate_forward_heading_source",
    "validate_motion_heading_correction",
    "validate_non_negative_parameter",
    "validate_positive_parameter",
    "validate_sidestep_heading_source",
    "validate_sidestep_smoothing",
    "validate_sidestep_suspect_mode",
]
