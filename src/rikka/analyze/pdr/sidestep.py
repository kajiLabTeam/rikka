"""旧運動状態・横歩き推定 import の互換 shim。"""
# ruff: noqa: F401

import sys
from typing import TYPE_CHECKING

from ...pdr.lib.motion_state import step_motion as _implementation

__all__ = [
    "_body_motion_angle_diff",
    "_circular_mean_angles",
    "_has_adjacent_opposite_evidence",
    "_heading_change_limit_for_movement_type",
    "_is_confirmed_sidestep_cluster",
    "_is_sidestep_bridge_gap",
    "_is_sidestep_movement",
    "_is_sidestep_suspect_movement",
    "_is_trajectory_sidestep_movement",
    "_lateral_forward_ratio",
    "_limit_heading_change",
    "_mean_finite",
    "_resolve_sidestep_heading",
    "_resolve_world_motion_heading",
    "_sidestep_body_lateral_heading",
    "_sidestep_cluster_bounds",
    "_sidestep_cluster_has_lateral_strength",
    "_sidestep_cluster_motion_heading",
    "_sidestep_cluster_motion_heading_for_indexes",
    "_sidestep_direction",
    "_sidestep_direction_label",
    "_sidestep_evidence",
    "_sidestep_motion_heading",
    "_SidestepEvidence",
    "_smooth_step_headings",
    "_smoothed_step_displacements",
    "_stabilize_trajectory_body_headings",
    "_stabilize_trajectory_headings",
    "_trajectory_movement_type",
    "build_particle_motion_headings",
    "build_step_motion_evidences",
    "build_step_motion_observations",
    "estimate_step_motion",
]

if TYPE_CHECKING:
    from ...pdr.lib.motion_state.step_motion import (
        _body_motion_angle_diff,
        _circular_mean_angles,
        _has_adjacent_opposite_evidence,
        _heading_change_limit_for_movement_type,
        _is_confirmed_sidestep_cluster,
        _is_sidestep_bridge_gap,
        _is_sidestep_movement,
        _is_sidestep_suspect_movement,
        _is_trajectory_sidestep_movement,
        _lateral_forward_ratio,
        _limit_heading_change,
        _mean_finite,
        _resolve_sidestep_heading,
        _resolve_world_motion_heading,
        _sidestep_body_lateral_heading,
        _sidestep_cluster_bounds,
        _sidestep_cluster_has_lateral_strength,
        _sidestep_cluster_motion_heading,
        _sidestep_cluster_motion_heading_for_indexes,
        _sidestep_direction,
        _sidestep_direction_label,
        _sidestep_evidence,
        _sidestep_motion_heading,
        _SidestepEvidence,
        _smooth_step_headings,
        _smoothed_step_displacements,
        _stabilize_trajectory_body_headings,
        _stabilize_trajectory_headings,
        _trajectory_movement_type,
        build_particle_motion_headings,
        build_step_motion_evidences,
        build_step_motion_observations,
        estimate_step_motion,
    )
else:
    sys.modules[__name__] = _implementation
