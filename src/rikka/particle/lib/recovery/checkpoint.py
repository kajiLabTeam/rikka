"""checkpoint から粒子列を再生する復旧戦略。

役割:
    健全だった過去状態から複数歩を再生し、壁を横切らない粒子列を選ぶ。
依存元:
    ``local`` の共通復旧結果、地図遷移判定、角度正規化を利用する。
利用先:
    particle の復旧段階と互換 API から呼び出される。
処理フロー:
    方位差と歩幅倍率を展開し、全再生歩が有効な候補を重み付き抽出する。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..map_constraints import _evaluate_particle_transitions
from ..proposal import _normalize_angle
from .local import _RecoveryResult


@dataclass(frozen=True)
class _CheckpointReplayResult:
    """checkpointから再生した粒子列と最終状態。"""

    recovery: _RecoveryResult
    replay_positions: np.ndarray


_REPLAY_PARAMETERS = (
    "checkpoint_particles",
    "checkpoint_heading_correction",
    "checkpoint_heading_drift",
    "checkpoint_stride_scale",
    "checkpoint_weights",
    "angles",
    "step_lengths",
    "n_particles",
    "map_gray",
    "gx_mean",
    "gz_mean",
    "origin_px",
    "scale",
    "heading_sigma",
    "rng",
    "checkpoint_motion_state",
    "allow_stride_adaptation",
    "stride_scale_min",
    "stride_scale_max",
    "capture_candidates",
)
_REPLAY_DEFAULTS = {
    "checkpoint_motion_state": None,
    "allow_stride_adaptation": False,
    "stride_scale_min": 0.5,
    "stride_scale_max": 1.6,
    "capture_candidates": False,
}


def _replay_from_checkpoint(
    *args: Any,
    **kwargs: Any,
) -> _CheckpointReplayResult | None:
    """同じ小方位差で最大3歩を再生し、壁非交差経路を返す。"""
    values = dict(zip(_REPLAY_PARAMETERS, args, strict=False))
    duplicated = set(values) & set(kwargs)
    if duplicated:
        raise TypeError(f"{sorted(duplicated)[0]} が重複指定されています")
    values.update(kwargs)
    for name, default in _REPLAY_DEFAULTS.items():
        values.setdefault(name, default)
    missing = [name for name in _REPLAY_PARAMETERS if name not in values]
    if missing:
        raise TypeError(f"必須引数が不足しています: {', '.join(missing)}")
    if len(args) > len(_REPLAY_PARAMETERS):
        raise TypeError("位置引数が多すぎます")

    checkpoint_particles = values["checkpoint_particles"]
    checkpoint_heading_correction = values["checkpoint_heading_correction"]
    checkpoint_heading_drift = values["checkpoint_heading_drift"]
    checkpoint_stride_scale = values["checkpoint_stride_scale"]
    checkpoint_weights = values["checkpoint_weights"]
    angles = values["angles"]
    step_lengths = values["step_lengths"]
    n_particles = values["n_particles"]
    map_gray = values["map_gray"]
    gx_mean = values["gx_mean"]
    gz_mean = values["gz_mean"]
    origin_px = values["origin_px"]
    scale = values["scale"]
    heading_sigma = values["heading_sigma"]
    rng = values["rng"]
    checkpoint_motion_state = values["checkpoint_motion_state"]
    allow_stride_adaptation = values["allow_stride_adaptation"]
    stride_scale_min = values["stride_scale_min"]
    stride_scale_max = values["stride_scale_max"]
    capture_candidates = values["capture_candidates"]
    if len(angles) == 0 or len(angles) != len(step_lengths):
        return None
    offset_degrees = np.array(
        [0.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 45.0, -45.0]
    )
    length_factors = (
        np.array([1.0, 0.85, 0.7, 1.15, 1.3])
        if allow_stride_adaptation
        else np.array([1.0, 0.9, 1.1])
    )
    combinations = np.array(
        [(offset, factor) for offset in offset_degrees for factor in length_factors]
    )
    parent_indices = np.repeat(np.arange(n_particles), len(combinations))
    combination_indices = np.tile(np.arange(len(combinations)), n_particles)
    batch_size = len(parent_indices)
    offsets = np.deg2rad(combinations[combination_indices, 0])
    length_factor = combinations[combination_indices, 1]
    parent_correction = checkpoint_heading_correction[parent_indices]
    parent_drift = checkpoint_heading_drift[parent_indices]
    candidate_stride_scales = checkpoint_stride_scale[parent_indices]
    current = checkpoint_particles[parent_indices].copy()
    replay_positions: list[np.ndarray] = []
    valid = np.ones(batch_size, dtype=bool)
    for angle, step_length in zip(angles, step_lengths, strict=True):
        theta = angle + parent_correction + parent_drift + offsets
        proposed = current.copy()
        effective_stride_scales = (
            length_factor
            if allow_stride_adaptation
            else candidate_stride_scales * length_factor
        )
        lengths = step_length * effective_stride_scales
        proposed[:, 0] += lengths * np.cos(theta)
        proposed[:, 1] += lengths * np.sin(theta)
        valid &= _evaluate_particle_transitions(
            current,
            proposed,
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        current = np.where(valid[:, None], proposed, current)
        replay_positions.append(current.copy())
    if not valid.any():
        return None

    valid &= checkpoint_weights[parent_indices] > 0.0
    if not valid.any():
        return None

    valid_indices = np.flatnonzero(valid)
    heading_cost = np.square(offsets[valid] / max(heading_sigma, np.deg2rad(5.0)))
    length_cost_sigma = 0.22 if allow_stride_adaptation else 0.1
    length_cost_center = (
        candidate_stride_scales[valid] if allow_stride_adaptation else 1.0
    )
    length_cost = np.square(
        (length_factor[valid] - length_cost_center) / length_cost_sigma
    )
    costs = heading_cost + length_cost
    probabilities = checkpoint_weights[parent_indices[valid]] * np.exp(
        -0.5 * (costs - costs.min())
    )
    probabilities /= probabilities.sum()
    selected_local = rng.choice(
        len(valid_indices),
        size=n_particles,
        replace=len(valid_indices) < n_particles,
        p=probabilities,
    )
    select = valid_indices[selected_local]
    selected_offsets = offsets[select]
    selected_factors = length_factor[select]
    selected_costs = costs[selected_local]
    recovery = _RecoveryResult(
        particles=current[select],
        heading_correction=parent_correction[select],
        heading_drift=_normalize_angle(parent_drift[select] + selected_offsets),
        stride_scale=np.clip(
            selected_factors
            if allow_stride_adaptation
            else candidate_stride_scales[select],
            stride_scale_min,
            stride_scale_max,
        ),
        motion_state=(
            np.zeros(n_particles, dtype=np.int8)
            if checkpoint_motion_state is None
            else checkpoint_motion_state[parent_indices[select]]
        ),
        parent_indices=parent_indices[select],
        valid_count=int(np.count_nonzero(valid)),
        attempts=1,
        mode="checkpoint_replay",
        heading_delta_deg=float(np.degrees(np.mean(np.abs(selected_offsets)))),
        step_scale=float(np.mean(selected_factors)),
        mean_cost=float(np.mean(selected_costs)),
        route_branch_ids=np.zeros(n_particles, dtype=np.int8),
        path_log_score_delta=-0.5 * selected_costs,
        candidate_headings=theta.copy() if capture_candidates else None,
        candidate_valid=valid.copy() if capture_candidates else None,
        selected_candidate_indices=select.copy() if capture_candidates else None,
    )
    selected_positions = np.stack(
        [positions[select] for positions in replay_positions],
        axis=0,
    )
    return _CheckpointReplayResult(recovery, selected_positions)
