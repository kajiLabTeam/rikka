"""マップ制約を使う粒子復旧とcheckpoint再生。

役割:
    通常伝播で有効粒子が不足した際の局所・旋回候補生成と、直近の健全な
    checkpointから複数歩を再生する復旧処理を提供する。
依存元:
    ``map_constraints`` の遷移判定、``motion`` の角度正規化、既存の
    ``particle_branches`` の枝保持再標本化、NumPyの配列演算と乱数生成器を使う。
利用先:
    粒子フィルタ実行処理が地図衝突後の粒子群と祖先情報を復元するために使用する。
処理フロー:
    方位差と歩幅倍率の候補を生成し、壁非交差候補を重み付き抽出する。局所候補が
    得られない場合に備え、checkpointから最大3歩の一貫した候補列も再構築する。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..branches import branch_preserving_resample
from ..map_constraints import _evaluate_particle_transitions
from ..proposal import _normalize_angle


@dataclass(frozen=True)
class _RecoveryResult:
    """保守的なmap-aware recoveryの内部結果。"""

    particles: np.ndarray
    heading_correction: np.ndarray
    heading_drift: np.ndarray
    stride_scale: np.ndarray
    motion_state: np.ndarray
    parent_indices: np.ndarray
    valid_count: int
    attempts: int
    mode: str
    heading_delta_deg: float
    step_scale: float
    mean_cost: float
    route_branch_ids: np.ndarray
    path_log_score_delta: np.ndarray
    candidate_headings: np.ndarray | None = None
    candidate_valid: np.ndarray | None = None
    selected_candidate_indices: np.ndarray | None = None


def _recovery_route_branch_ids(offsets: np.ndarray) -> np.ndarray:
    """recovery方位差を直進・左・右・後退の経路族へ分類する。"""
    degrees = np.degrees(_normalize_angle(offsets))
    absolute_degrees = np.abs(degrees)
    branch_ids = np.zeros(len(offsets), dtype=np.int8)
    branch_ids[(degrees >= 60.0) & (absolute_degrees < 135.0)] = 1
    branch_ids[(degrees <= -60.0) & (absolute_degrees < 135.0)] = 2
    branch_ids[absolute_degrees >= 135.0] = 3
    return branch_ids


_RECOVERY_PARAMETERS = (
    "previous_particles",
    "previous_heading_correction",
    "previous_heading_drift",
    "previous_stride_scale",
    "proposed_motion_state",
    "previous_weights",
    "angle_det",
    "step_length",
    "sigma_step_length_ratio",
    "n_particles",
    "map_gray",
    "gx_mean",
    "gz_mean",
    "origin_px",
    "scale",
    "heading_sigma",
    "max_attempts",
    "rng",
    "allow_turn_candidates",
    "preserve_route_branches",
    "allow_stride_adaptation",
    "stride_scale_min",
    "stride_scale_max",
    "capture_candidates",
)
_RECOVERY_DEFAULTS = {
    "allow_turn_candidates": False,
    "preserve_route_branches": True,
    "allow_stride_adaptation": False,
    "stride_scale_min": 0.5,
    "stride_scale_max": 1.6,
    "capture_candidates": False,
}


def _generate_recovery_candidates(
    *args: Any,
    **kwargs: Any,
) -> _RecoveryResult | None:
    """決定論的方位に近い壁非交差候補から復旧粒子を生成する。"""
    values = dict(zip(_RECOVERY_PARAMETERS, args, strict=False))
    duplicated = set(values) & set(kwargs)
    if duplicated:
        raise TypeError(f"{sorted(duplicated)[0]} が重複指定されています")
    values.update(kwargs)
    for name, default in _RECOVERY_DEFAULTS.items():
        values.setdefault(name, default)
    missing = [name for name in _RECOVERY_PARAMETERS if name not in values]
    if missing:
        raise TypeError(f"必須引数が不足しています: {', '.join(missing)}")
    if len(args) > len(_RECOVERY_PARAMETERS):
        raise TypeError("位置引数が多すぎます")

    previous_particles = values["previous_particles"]
    previous_heading_correction = values["previous_heading_correction"]
    previous_heading_drift = values["previous_heading_drift"]
    previous_stride_scale = values["previous_stride_scale"]
    proposed_motion_state = values["proposed_motion_state"]
    previous_weights = values["previous_weights"]
    angle_det = values["angle_det"]
    step_length = values["step_length"]
    sigma_step_length_ratio = values["sigma_step_length_ratio"]
    n_particles = values["n_particles"]
    map_gray = values["map_gray"]
    gx_mean = values["gx_mean"]
    gz_mean = values["gz_mean"]
    origin_px = values["origin_px"]
    scale = values["scale"]
    heading_sigma = values["heading_sigma"]
    max_attempts = values["max_attempts"]
    rng = values["rng"]
    allow_turn_candidates = values["allow_turn_candidates"]
    preserve_route_branches = values["preserve_route_branches"]
    allow_stride_adaptation = values["allow_stride_adaptation"]
    stride_scale_min = values["stride_scale_min"]
    stride_scale_max = values["stride_scale_max"]
    capture_candidates = values["capture_candidates"]
    local_degrees = np.array(
        [0.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 45.0, -45.0]
    )
    normal_length_factors = (
        np.array([1.0, 0.85, 0.7, 1.15, 1.3])
        if allow_stride_adaptation
        else np.array([1.0, 0.9, 1.1])
    )
    stages = [("local_grid", local_degrees, normal_length_factors)]
    if allow_turn_candidates:
        stages.append(
            (
                "turn_grid",
                np.array([60.0, -60.0, 90.0, -90.0, 135.0, -135.0, 180.0]),
                normal_length_factors,
            )
        )

    candidate_particles: list[np.ndarray] = []
    candidate_parent_indices: list[np.ndarray] = []
    candidate_parent_corrections: list[np.ndarray] = []
    candidate_parent_drifts: list[np.ndarray] = []
    candidate_stride_scales_list: list[np.ndarray] = []
    candidate_offsets: list[np.ndarray] = []
    candidate_length_factors: list[np.ndarray] = []
    candidate_costs: list[np.ndarray] = []
    captured_headings: list[np.ndarray] = []
    captured_valid: list[np.ndarray] = []
    captured_valid_indices: list[np.ndarray] = []
    captured_count = 0
    attempts = 0
    for stage_number, (_mode, offset_degrees, length_factors) in enumerate(
        stages[:max_attempts], start=1
    ):
        attempts = stage_number
        combinations = np.array(
            [(offset, factor) for offset in offset_degrees for factor in length_factors]
        )
        parent_indices = np.repeat(np.arange(n_particles), len(combinations))
        combination_indices = np.tile(np.arange(len(combinations)), n_particles)
        batch_size = len(parent_indices)
        offsets = np.deg2rad(combinations[combination_indices, 0])
        length_factor = combinations[combination_indices, 1]
        starts = previous_particles[parent_indices]
        parent_correction = previous_heading_correction[parent_indices]
        parent_drift = previous_heading_drift[parent_indices]
        candidate_stride_scales = previous_stride_scale[parent_indices]
        base_angles = np.asarray(angle_det, dtype=float)
        parent_angles = (
            np.full(batch_size, float(base_angles))
            if base_angles.ndim == 0
            else base_angles[parent_indices]
        )
        theta = parent_angles + parent_correction + parent_drift + offsets
        residual_noise = rng.normal(
            0.0,
            sigma_step_length_ratio,
            batch_size,
        )
        base_lengths = np.asarray(step_length, dtype=float)
        parent_lengths = (
            np.full(batch_size, float(base_lengths))
            if base_lengths.ndim == 0
            else base_lengths[parent_indices]
        )
        effective_stride_scales = (
            length_factor
            if allow_stride_adaptation
            else candidate_stride_scales * length_factor
        )
        lengths = np.clip(
            parent_lengths * effective_stride_scales * (1.0 + residual_noise),
            0.0,
            None,
        )
        candidates = starts.copy()
        candidates[:, 0] += lengths * np.cos(theta)
        candidates[:, 1] += lengths * np.sin(theta)
        valid = _evaluate_particle_transitions(
            starts,
            candidates,
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        valid &= previous_weights[parent_indices] > 0.0
        if capture_candidates:
            captured_headings.append(theta.copy())
            captured_valid.append(valid.copy())
            captured_valid_indices.append(np.flatnonzero(valid) + captured_count)
            captured_count += len(theta)
        if not valid.any():
            continue

        heading_cost = np.square(offsets[valid] / max(heading_sigma, np.deg2rad(5.0)))
        length_cost_sigma = 0.22 if allow_stride_adaptation else 0.1
        length_cost_center = (
            candidate_stride_scales[valid] if allow_stride_adaptation else 1.0
        )
        length_cost = np.square(
            (length_factor[valid] - length_cost_center) / length_cost_sigma
        )
        costs = heading_cost + length_cost
        if not preserve_route_branches:
            valid_indexes = np.flatnonzero(valid)
            probabilities = previous_weights[parent_indices[valid]] * np.exp(
                -0.5 * (costs - costs.min())
            )
            probabilities /= probabilities.sum()
            selected_local = rng.choice(
                len(valid_indexes),
                size=n_particles,
                replace=len(valid_indexes) < n_particles,
                p=probabilities,
            )
            select = valid_indexes[selected_local]
            selected_offsets = offsets[select]
            selected_factors = length_factor[select]
            return _RecoveryResult(
                particles=candidates[select],
                heading_correction=parent_correction[select],
                heading_drift=_normalize_angle(parent_drift[select] + selected_offsets),
                stride_scale=np.clip(
                    selected_factors
                    if allow_stride_adaptation
                    else candidate_stride_scales[select],
                    stride_scale_min,
                    stride_scale_max,
                ),
                motion_state=proposed_motion_state[parent_indices[select]],
                parent_indices=parent_indices[select],
                valid_count=int(np.count_nonzero(valid)),
                attempts=stage_number,
                mode=_mode,
                heading_delta_deg=float(np.degrees(np.mean(np.abs(selected_offsets)))),
                step_scale=float(np.mean(selected_factors)),
                mean_cost=float(np.mean(costs[selected_local])),
                route_branch_ids=np.zeros(n_particles, dtype=np.int8),
                path_log_score_delta=-0.5 * costs[selected_local],
                candidate_headings=theta.copy() if capture_candidates else None,
                candidate_valid=valid.copy() if capture_candidates else None,
                selected_candidate_indices=(
                    select.copy() if capture_candidates else None
                ),
            )
        candidate_particles.append(candidates[valid])
        candidate_parent_indices.append(parent_indices[valid])
        candidate_parent_corrections.append(parent_correction[valid])
        candidate_parent_drifts.append(parent_drift[valid])
        candidate_stride_scales_list.append(candidate_stride_scales[valid])
        candidate_offsets.append(offsets[valid])
        candidate_length_factors.append(length_factor[valid])
        candidate_costs.append(costs)

        # turn候補を使わない従来経路では、local候補が見つかれば次段階はない。
        if not allow_turn_candidates:
            break

    if not candidate_particles:
        return None

    particles_all = np.concatenate(candidate_particles)
    parents_all = np.concatenate(candidate_parent_indices)
    corrections_all = np.concatenate(candidate_parent_corrections)
    drifts_all = np.concatenate(candidate_parent_drifts)
    stride_scales_all = np.concatenate(candidate_stride_scales_list)
    offsets_all = np.concatenate(candidate_offsets)
    length_factors_all = np.concatenate(candidate_length_factors)
    costs_all = np.concatenate(candidate_costs)
    route_branch_ids = _recovery_route_branch_ids(offsets_all)
    log_probabilities = np.log(previous_weights[parents_all]) - 0.5 * costs_all
    relative_log_probabilities = np.clip(
        log_probabilities - float(np.max(log_probabilities)),
        -700.0,
        0.0,
    )
    probabilities = np.exp(relative_log_probabilities)
    probabilities /= probabilities.sum()
    select, selected_route_branch_ids, _branch_diagnostics = branch_preserving_resample(
        probabilities,
        route_branch_ids,
        rng,
        output_count=n_particles,
    )
    selected_offsets = offsets_all[select]
    selected_factors = length_factors_all[select]
    selected_costs = costs_all[select]
    selected_correction = corrections_all[select]
    selected_drift = _normalize_angle(drifts_all[select] + selected_offsets)
    selected_candidate_indices = (
        np.concatenate(captured_valid_indices)[select] if capture_candidates else None
    )
    return _RecoveryResult(
        particles=particles_all[select],
        heading_correction=selected_correction,
        heading_drift=selected_drift,
        stride_scale=np.clip(
            selected_factors if allow_stride_adaptation else stride_scales_all[select],
            stride_scale_min,
            stride_scale_max,
        ),
        motion_state=proposed_motion_state[parents_all[select]],
        parent_indices=parents_all[select],
        valid_count=len(particles_all),
        attempts=attempts,
        mode=("turn_grid" if np.any(selected_route_branch_ids != 0) else "local_grid"),
        heading_delta_deg=float(np.degrees(np.mean(np.abs(selected_offsets)))),
        step_scale=float(np.mean(selected_factors)),
        mean_cost=float(np.mean(selected_costs)),
        route_branch_ids=selected_route_branch_ids,
        path_log_score_delta=-0.5 * selected_costs,
        candidate_headings=(
            np.concatenate(captured_headings) if capture_candidates else None
        ),
        candidate_valid=(
            np.concatenate(captured_valid) if capture_candidates else None
        ),
        selected_candidate_indices=selected_candidate_indices,
    )
