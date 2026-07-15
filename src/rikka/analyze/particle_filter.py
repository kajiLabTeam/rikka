"""パーティクルフィルタによる確率的歩行軌跡推定。

役割:
    PDR の歩行ステップへ確率的な方位・歩幅ノイズとフロアマップ制約を適用し、
    マップマッチング済み軌跡、静止画、アニメーションを生成する。
依存元:
    ``config`` から既定値、``pdr.particle_api`` から方位・歩幅・横歩き判定・描画の
    共有 API を取得し、NumPy、Pandas、SciPy、Matplotlib を数値処理と可視化に使う。
利用先:
    ``pdr.pipeline.run`` が particle モードで遅延 import し、CLI の ``particle``
    コマンドおよび particle 有効時の ``run`` から使用する。
処理フロー:
    ステップ候補と永続歩幅倍率を粒子群へ反映し、全通過画素の壁制約で重み付け、
    必要時の系統リサンプリングと復旧、祖先経路復元を行い、重み付き平均を
    優先しつつ壁付近だけ粒子祖先へ退避する到達可能な軌跡を返す。
"""

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.ndimage import distance_transform_edt

from ..config import (
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    INITIAL_DIRECTION,
    PF_HEADING_DRIFT_RETENTION,
    PF_MOTION_STATE_TRANSITION_STAY,
    PF_NUM_PARTICLES,
    PF_RECOVERY_HEADING_SIGMA,
    PF_RECOVERY_MAX_ATTEMPTS,
    PF_RECOVERY_VALID_RATIO,
    PF_REJUVENATION_SIGMA_HEADING,
    PF_RESAMPLE_ESS_RATIO,
    PF_SIGMA_HEADING,
    PF_SIGMA_INIT_HEADING,
    PF_SIGMA_STEP_LENGTH_RATIO,
    PF_STRIDE_SCALE_INIT_SIGMA,
    PF_STRIDE_SCALE_MAX,
    PF_STRIDE_SCALE_MIN,
    PF_STRIDE_SCALE_PRIOR_MEAN,
    PF_STRIDE_SCALE_PROCESS_SIGMA,
    PF_STRIDE_SCALE_REJUVENATION_SIGMA,
    PF_STRIDE_SCALE_RETENTION,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_LENGTH_SCALE,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    SIDESTEP_SUSPECT_MODE,
    STEP_LENGTH_METHOD,
    TURNING_LENGTH_SCALE,
    WEINBERG_K,
)
from .pdr.particle_api import (
    StepHeading,
    StepMotionEvidence,
    StepSegment,
    build_particle_motion_headings,
    build_step_motion_evidences,
    compute_pixel_coords,
    estimate_device_orientation_mode,
    estimate_initial_forward_angle,
    estimate_step_length,
    estimate_step_length_forward,
    estimate_step_motion,
    plot_heading_overlay,
    resolve_motion_heading_correction,
    resolve_step_heading,
    smooth_step_headings,
    stabilize_trajectory_headings,
    step_output_time,
    validate_forward_heading_source,
    validate_motion_heading_correction,
    validate_non_negative_parameter,
    validate_positive_parameter,
    validate_sidestep_heading_source,
    validate_sidestep_smoothing,
    validate_sidestep_suspect_mode,
)

_MOTION_FORWARD = 0
_MOTION_SIDESTEP_LEFT = 1
_MOTION_SIDESTEP_RIGHT = 2
_MOTION_TURNING = 3
_MOTION_STATE_NAMES = ("forward", "sidestep_left", "sidestep_right", "turning")


@dataclass(frozen=True)
class ParticleFilterStepDiagnostics:
    """1歩分の粒子健全性と復旧結果。"""

    step: int
    timestamp_s: float
    valid_count: int
    valid_ratio: float
    valid_weight_count: int
    valid_weight_mass_before_normalization: float
    ess_before_observation: float
    ess_after_observation: float
    ess_after_resampling: float
    max_weight: float
    unique_parent_count: int
    unique_position_count: int
    position_spread_rms_m: float
    heading_drift_std_deg: float
    heading_total_std_deg: float
    stride_scale_mean: float
    stride_scale_std: float
    effective_step_length_mean_m: float
    effective_step_length_std_m: float
    forward_state_probability: float
    sidestep_left_state_probability: float
    sidestep_right_state_probability: float
    turning_state_probability: float
    representative_motion_state: str
    motion_state_entropy: float
    motion_state_transition_count: int
    motion_reliability: float
    calibration_reliability: float
    resampled: bool
    recovery_attempted: bool
    recovery_mode: str
    recovery_valid_count: int
    recovery_attempts: int
    recovery_heading_delta_deg: float | None
    recovery_step_scale: float | None
    recovery_cost: float | None
    recovery_checkpoint_step: int | None
    recovery_replay_steps: int
    trajectory_mode: str
    trajectory_source_index: int | None


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """系統リサンプリングでインデックス配列を返す。O(N)・分散最小。"""
    n = len(weights)
    positions = (np.arange(n) + rng.uniform(0, 1)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


def _effective_sample_size(weights: np.ndarray) -> float:
    """正規化済み重みから有効サンプルサイズを返す。"""
    squared_sum = float(np.sum(np.square(weights)))
    if squared_sum <= 0.0 or not np.isfinite(squared_sum):
        return 0.0
    return 1.0 / squared_sum


def _normalize_angle(angle: np.ndarray) -> np.ndarray:
    """角度配列を -pi 以上 pi 未満へ正規化する。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _motion_state_transition_matrix() -> np.ndarray:
    """前進・左右横歩き・旋回の継続性を表す遷移行列を返す。"""
    side_stay = PF_MOTION_STATE_TRANSITION_STAY
    return np.asarray(
        [
            [0.90, 0.035, 0.035, 0.03],
            [0.12, side_stay, 0.01, 0.05],
            [0.12, 0.01, side_stay, 0.05],
            [0.45, 0.08, 0.08, 0.39],
        ],
        dtype=float,
    )


def _sample_motion_states(
    previous_states: np.ndarray,
    observation_likelihoods: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """遷移と観測を使う最適提案分布から次状態と予測尤度を返す。"""
    unnormalized = (
        _motion_state_transition_matrix()[previous_states]
        * observation_likelihoods[None, :]
    )
    predictive_likelihoods = unnormalized.sum(axis=1)
    probabilities = unnormalized / predictive_likelihoods[:, None]
    draws = rng.random(len(previous_states))
    states = np.sum(draws[:, None] > np.cumsum(probabilities, axis=1), axis=1).astype(
        np.int8
    )
    return states, predictive_likelihoods


def _motion_state_headings(
    step_heading: StepHeading,
    evidence: StepMotionEvidence,
    particle_heading: float,
) -> np.ndarray:
    """1歩の運動状態別方位候補を返す。"""
    fallback = particle_heading
    if (
        evidence.calibration_reliability >= 0.85
        or step_heading.sidestep_cluster_id is None
    ):
        return np.full(4, fallback, dtype=float)
    body = (
        step_heading.body_heading if step_heading.body_heading is not None else fallback
    )
    movement = step_heading.trajectory_movement_type or step_heading.movement_type
    motion = (
        step_heading.motion_heading
        if step_heading.motion_heading is not None
        else fallback
    )
    forward = body
    left = body + np.pi / 2
    right = body - np.pi / 2
    if movement in {
        "sidestep_left",
        "turning_sidestep_left",
        "sidestep_suspect_left",
    }:
        left = motion
    elif movement in {
        "sidestep_right",
        "turning_sidestep_right",
        "sidestep_suspect_right",
    }:
        right = motion
    turning = motion if movement.startswith("turning") else fallback
    return _normalize_angle(np.asarray([forward, left, right, turning], dtype=float))


def _motion_state_likelihoods(evidence: StepMotionEvidence) -> np.ndarray:
    """運動観測を粒子状態順の尤度配列へ変換する。"""
    return np.asarray(
        [
            evidence.forward_likelihood,
            evidence.sidestep_left_likelihood,
            evidence.sidestep_right_likelihood,
            evidence.turning_likelihood,
        ],
        dtype=float,
    )


def _weighted_circular_std(angles: np.ndarray, weights: np.ndarray) -> float:
    """重み付き角度分布の円周標準偏差をラジアンで返す。"""
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    cosine = float(np.sum(weights * np.cos(angles)) / total)
    sine = float(np.sum(weights * np.sin(angles)) / total)
    resultant = float(np.clip(np.hypot(cosine, sine), 1e-15, 1.0))
    return float(np.sqrt(max(0.0, -2.0 * np.log(resultant))))


def _normalize_floormap_gray(map_raw: np.ndarray) -> np.ndarray:
    """フロアマップ画像を 0..255 のグレースケール配列に正規化する。"""
    map_arr: np.ndarray = np.asarray(map_raw, dtype=float)
    if map_arr.ndim == 3:
        map_arr = np.mean(map_arr[:, :, :3], axis=2)
    if map_arr.size == 0:
        return map_arr
    if float(np.nanmax(map_arr)) <= 1.0:
        map_arr = map_arr * 255.0
    return np.asarray(np.clip(map_arr, 0.0, 255.0), dtype=float)


def _pixel_y_sign(gx_mean: float, gz_mean: float) -> int:
    """メートル座標とピクセル座標のY軸向きを返す。"""
    if abs(gx_mean) > abs(gz_mean):
        return -1 if gx_mean > 0 else 1
    return -1 if gz_mean < 0 else 1


def _compute_meter_coords(
    px: np.ndarray,
    py: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """フロアマップのピクセル座標をメートル座標へ戻す。"""
    y_sign = _pixel_y_sign(gx_mean, gz_mean)
    xs = (px - origin_px[0]) * scale
    ys = (py - origin_px[1]) * scale / y_sign
    return xs, ys


def _is_walkable_cell(
    map_gray: np.ndarray,
    x: int,
    y: int,
) -> bool:
    """指定画素がマップ内の歩行可能画素かを返す。"""
    map_h, map_w = map_gray.shape
    return 0 <= x < map_w and 0 <= y < map_h and bool(map_gray[y, x] > 128)


def _segment_crosses_only_walkable_cells(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    map_gray: np.ndarray,
) -> bool:
    """線分が触れる全画素を保守的に調べ、歩行可能かを返す。"""
    if not np.isfinite([x0, y0, x1, y1]).all():
        return False

    cell_x = int(np.floor(x0 + 0.5))
    cell_y = int(np.floor(y0 + 0.5))
    end_x = int(np.floor(x1 + 0.5))
    end_y = int(np.floor(y1 + 0.5))
    if not _is_walkable_cell(map_gray, cell_x, cell_y):
        return False
    if cell_x == end_x and cell_y == end_y:
        return True

    dx = x1 - x0
    dy = y1 - y0
    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_y = 1 if dy > 0 else -1 if dy < 0 else 0
    t_delta_x = np.inf if step_x == 0 else 1.0 / abs(dx)
    t_delta_y = np.inf if step_y == 0 else 1.0 / abs(dy)
    next_boundary_x = cell_x + 0.5 if step_x > 0 else cell_x - 0.5
    next_boundary_y = cell_y + 0.5 if step_y > 0 else cell_y - 0.5
    t_max_x = np.inf if step_x == 0 else (next_boundary_x - x0) / dx
    t_max_y = np.inf if step_y == 0 else (next_boundary_y - y0) / dy

    max_cells = abs(end_x - cell_x) + abs(end_y - cell_y) + 2
    for _ in range(max_cells):
        if cell_x == end_x and cell_y == end_y:
            return True
        if abs(t_max_x - t_max_y) <= 1e-12:
            next_x = cell_x + step_x
            next_y = cell_y + step_y
            # 画素角に触れる遷移は、隣接する両画素も通過したものとして扱う。
            if not _is_walkable_cell(map_gray, next_x, cell_y):
                return False
            if not _is_walkable_cell(map_gray, cell_x, next_y):
                return False
            cell_x = next_x
            cell_y = next_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            cell_x += step_x
            t_max_x += t_delta_x
        else:
            cell_y += step_y
            t_max_y += t_delta_y
        if not _is_walkable_cell(map_gray, cell_x, cell_y):
            return False
    return False


def _evaluate_particle_transitions(
    previous_particles: np.ndarray,
    proposed_particles: np.ndarray,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> np.ndarray:
    """各粒子の移動線分が全て歩行可能画素だけを通るか判定する。"""
    if previous_particles.shape != proposed_particles.shape:
        raise ValueError("遷移前後の粒子配列 shape が一致しません")
    previous_px, previous_py = compute_pixel_coords(
        previous_particles[:, 0],
        previous_particles[:, 1],
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    proposed_px, proposed_py = compute_pixel_coords(
        proposed_particles[:, 0],
        proposed_particles[:, 1],
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    return np.asarray(
        [
            _segment_crosses_only_walkable_cells(x0, y0, x1, y1, map_gray)
            for x0, y0, x1, y1 in zip(
                previous_px,
                previous_py,
                proposed_px,
                proposed_py,
                strict=True,
            )
        ],
        dtype=bool,
    )


def _snap_trajectory_to_walkable_pixels(
    trajectory: list[list[float]],
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> list[list[float]]:
    """壁上・範囲外の軌跡点を最近傍の歩行可能画素へ寄せる。"""
    if len(trajectory) == 0:
        return trajectory

    walkable = map_gray > 128
    if not walkable.any():
        return trajectory

    map_h, map_w = walkable.shape
    points = np.asarray(trajectory, dtype=float)
    px_f, py_f = compute_pixel_coords(
        points[:, 0], points[:, 1], gx_mean, gz_mean, origin_px, scale
    )
    pxi = np.round(px_f).astype(int)
    pyi = np.round(py_f).astype(int)

    in_bounds = (0 <= pxi) & (pxi < map_w) & (0 <= pyi) & (pyi < map_h)
    needs_snap = ~in_bounds.copy()
    if in_bounds.any():
        needs_snap[in_bounds] = ~walkable[pyi[in_bounds], pxi[in_bounds]]

    if not needs_snap.any():
        return trajectory

    _, nearest_indices = distance_transform_edt(~walkable, return_indices=True)
    nearest_y = nearest_indices[0]
    nearest_x = nearest_indices[1]

    query_x = pxi.clip(0, map_w - 1)
    query_y = pyi.clip(0, map_h - 1)
    snap_x = query_x.copy()
    snap_y = query_y.copy()
    snap_x[needs_snap] = nearest_x[query_y[needs_snap], query_x[needs_snap]]
    snap_y[needs_snap] = nearest_y[query_y[needs_snap], query_x[needs_snap]]

    xs, ys = _compute_meter_coords(
        snap_x.astype(float),
        snap_y.astype(float),
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    snapped_points = points.copy()
    snapped_points[needs_snap, 0] = xs[needs_snap]
    snapped_points[needs_snap, 1] = ys[needs_snap]
    return [[float(x), float(y)] for x, y in snapped_points]


def _reconstruct_resampled_paths(
    position_history: list[np.ndarray],
    resample_history: list[np.ndarray],
) -> np.ndarray:
    """リサンプリング祖先をたどって最終粒子群の経路を復元する。"""
    if not position_history:
        return np.empty((0, 0, 2), dtype=float)

    n_particles = position_history[0].shape[0]
    n_steps = len(position_history) - 1
    if len(resample_history) != n_steps:
        raise ValueError("position_history と resample_history の長さが一致しません")

    paths = np.empty((n_particles, n_steps + 1, 2), dtype=float)
    if n_steps == 0:
        paths[:, 0, :] = position_history[0]
        return paths

    lineage = resample_history[-1].astype(int, copy=True)
    paths[:, n_steps, :] = position_history[n_steps][lineage]

    for step in range(n_steps - 1, 0, -1):
        lineage = resample_history[step - 1][lineage]
        paths[:, step, :] = position_history[step][lineage]

    paths[:, 0, :] = position_history[0][lineage]
    return paths


def _reconstruct_particle_paths(
    position_history: list[np.ndarray],
    parent_history: list[np.ndarray],
) -> np.ndarray:
    """各ステップ後の親インデックスから最終粒子の有効経路を復元する。"""
    if not position_history:
        return np.empty((0, 0, 2), dtype=float)
    n_particles = position_history[0].shape[0]
    n_steps = len(position_history) - 1
    if len(parent_history) != n_steps:
        raise ValueError("position_history と parent_history の長さが一致しません")

    paths = np.empty((n_particles, n_steps + 1, 2), dtype=float)
    lineage = np.arange(n_particles, dtype=int)
    for step in range(n_steps, 0, -1):
        paths[:, step, :] = position_history[step][lineage]
        lineage = parent_history[step - 1][lineage]
    paths[:, 0, :] = position_history[0][lineage]
    return paths


def _select_reachable_mean_path(
    particle_paths: np.ndarray,
    final_weights: np.ndarray,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, list[str], list[int | None]]:
    """平均を優先し、壁区間だけ同じ粒子祖先へ退避した経路を返す。"""
    positive_indices = np.flatnonzero(final_weights > 0.0)
    if positive_indices.size == 0:
        raise RuntimeError("正の最終重みを持つ粒子がありません")

    positive_paths = particle_paths[positive_indices]
    positive_weights = final_weights[positive_indices].astype(float, copy=True)
    positive_weights /= positive_weights.sum()
    mean_path = np.average(positive_paths, axis=0, weights=positive_weights)

    n_paths, n_times, _ = positive_paths.shape
    n_states = n_paths + 1  # 0は平均、1以降は同じ祖先を追う粒子状態
    costs = np.zeros(n_states, dtype=float)
    backpointers = np.full((n_times, n_states), -1, dtype=int)

    for time_index in range(1, n_times):
        next_costs = np.full(n_states, np.inf, dtype=float)

        mean_to_mean = bool(
            _evaluate_particle_transitions(
                mean_path[time_index - 1 : time_index],
                mean_path[time_index : time_index + 1],
                map_gray,
                gx_mean,
                gz_mean,
                origin_px,
                scale,
            )[0]
        )
        if mean_to_mean and np.isfinite(costs[0]):
            next_costs[0] = costs[0]
            backpointers[time_index, 0] = 0

        particle_to_mean = _evaluate_particle_transitions(
            positive_paths[:, time_index - 1, :],
            np.repeat(mean_path[None, time_index, :], n_paths, axis=0),
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        mean_predecessor_costs = np.where(particle_to_mean, costs[1:], np.inf)
        best_particle = int(np.argmin(mean_predecessor_costs))
        if mean_predecessor_costs[best_particle] < next_costs[0]:
            next_costs[0] = mean_predecessor_costs[best_particle]
            backpointers[time_index, 0] = best_particle + 1

        deviations = np.sum(
            np.square(positive_paths[:, time_index, :] - mean_path[time_index]),
            axis=1,
        )
        # 同じ祖先を追い続ける遷移は、PF伝播時に壁判定済みである。
        same_path_costs = costs[1:] + deviations
        next_costs[1:] = same_path_costs
        backpointers[time_index, 1:] = np.arange(1, n_states)

        mean_to_particle = _evaluate_particle_transitions(
            np.repeat(mean_path[None, time_index - 1, :], n_paths, axis=0),
            positive_paths[:, time_index, :],
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        enter_costs = np.where(mean_to_particle, costs[0] + deviations, np.inf)
        use_mean_predecessor = enter_costs < next_costs[1:]
        particle_state_indices = np.flatnonzero(use_mean_predecessor) + 1
        next_costs[particle_state_indices] = enter_costs[use_mean_predecessor]
        backpointers[time_index, particle_state_indices] = 0

        if not np.isfinite(next_costs).any():
            raise RuntimeError("到達可能な平均・粒子経路を構成できません")
        costs = next_costs

    state = int(np.argmin(costs))
    states = np.empty(n_times, dtype=int)
    states[-1] = state
    for time_index in range(n_times - 1, 0, -1):
        state = int(backpointers[time_index, state])
        if state < 0:
            raise RuntimeError("軌跡の親状態を復元できません")
        states[time_index - 1] = state

    selected_path = np.empty_like(mean_path)
    modes: list[str] = []
    sources: list[int | None] = []
    for time_index, selected_state in enumerate(states):
        if selected_state == 0:
            selected_path[time_index] = mean_path[time_index]
            modes.append("weighted_mean")
            sources.append(None)
        else:
            local_index = selected_state - 1
            selected_path[time_index] = positive_paths[local_index, time_index]
            modes.append("particle_fallback")
            sources.append(int(positive_indices[local_index]))

    if (
        n_times > 1
        and not _evaluate_particle_transitions(
            selected_path[:-1],
            selected_path[1:],
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        ).all()
    ):
        raise RuntimeError("構成した軌跡に壁またはマップ外遷移が含まれます")
    return selected_path, modes, sources


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


@dataclass(frozen=True)
class _CheckpointReplayResult:
    """checkpointから再生した粒子列と最終状態。"""

    recovery: _RecoveryResult
    replay_positions: np.ndarray


def _generate_recovery_candidates(
    previous_particles: np.ndarray,
    previous_heading_correction: np.ndarray,
    previous_heading_drift: np.ndarray,
    previous_stride_scale: np.ndarray,
    proposed_motion_state: np.ndarray,
    previous_weights: np.ndarray,
    angle_det: float | np.ndarray,
    step_length: float | np.ndarray,
    sigma_step_length_ratio: float,
    n_particles: int,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
    heading_sigma: float,
    max_attempts: int,
    rng: np.random.Generator,
    allow_turn_candidates: bool = False,
) -> _RecoveryResult | None:
    """決定論的方位に近い壁非交差候補から復旧粒子を生成する。"""
    local_degrees = np.array(
        [0.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 45.0, -45.0]
    )
    normal_length_factors = np.array([1.0, 0.9, 1.1])
    stages = [("local_grid", local_degrees, normal_length_factors)]
    if allow_turn_candidates:
        stages.append(
            (
                "turn_grid",
                np.array([60.0, -60.0, 90.0, -90.0, 135.0, -135.0, 180.0]),
                normal_length_factors,
            )
        )

    for stage_number, (mode, offset_degrees, length_factors) in enumerate(
        stages[:max_attempts], start=1
    ):
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
        lengths = np.clip(
            parent_lengths
            * candidate_stride_scales
            * length_factor
            * (1.0 + residual_noise),
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
        if not valid.any():
            continue

        valid_indices = np.flatnonzero(valid)
        heading_cost = np.square(offsets[valid] / max(heading_sigma, np.deg2rad(5.0)))
        length_cost = np.square((length_factor[valid] - 1.0) / 0.1)
        costs = heading_cost + length_cost
        probabilities = previous_weights[parent_indices[valid]] * np.exp(
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
        selected_costs = heading_cost[selected_local] + length_cost[selected_local]
        selected_correction = parent_correction[select]
        selected_drift = _normalize_angle(parent_drift[select] + selected_offsets)
        return _RecoveryResult(
            particles=candidates[select],
            heading_correction=selected_correction,
            heading_drift=selected_drift,
            stride_scale=candidate_stride_scales[select],
            motion_state=proposed_motion_state[parent_indices[select]],
            parent_indices=parent_indices[select],
            valid_count=int(np.count_nonzero(valid)),
            attempts=stage_number,
            mode=mode,
            heading_delta_deg=float(np.degrees(np.mean(np.abs(selected_offsets)))),
            step_scale=float(np.mean(selected_factors)),
            mean_cost=float(np.mean(selected_costs)),
        )

    return None


def _replay_from_checkpoint(
    checkpoint_particles: np.ndarray,
    checkpoint_heading_correction: np.ndarray,
    checkpoint_heading_drift: np.ndarray,
    checkpoint_stride_scale: np.ndarray,
    checkpoint_weights: np.ndarray,
    angles: np.ndarray,
    step_lengths: np.ndarray,
    n_particles: int,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
    heading_sigma: float,
    rng: np.random.Generator,
    checkpoint_motion_state: np.ndarray | None = None,
) -> _CheckpointReplayResult | None:
    """同じ小方位差で最大3歩を再生し、壁非交差経路を返す。"""
    if len(angles) == 0 or len(angles) != len(step_lengths):
        return None
    offset_degrees = np.array(
        [0.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 45.0, -45.0]
    )
    length_factors = np.array([1.0, 0.9, 1.1])
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
        lengths = step_length * candidate_stride_scales * length_factor
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
    length_cost = np.square((length_factor[valid] - 1.0) / 0.1)
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
        stride_scale=candidate_stride_scales[select],
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
    )
    selected_positions = np.stack(
        [positions[select] for positions in replay_positions],
        axis=0,
    )
    return _CheckpointReplayResult(
        recovery=recovery,
        replay_positions=selected_positions,
    )


def run_particle_filter(
    peaks: np.ndarray,
    df_gyro: pd.DataFrame,
    df_acc: pd.DataFrame,
    gx_mean: float,
    gz_mean: float,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    initial_direction: float = INITIAL_DIRECTION,
    n_particles: int = PF_NUM_PARTICLES,
    sigma_init_heading: float = PF_SIGMA_INIT_HEADING,
    sigma_heading: float = PF_SIGMA_HEADING,
    sigma_sl_ratio: float = PF_SIGMA_STEP_LENGTH_RATIO,
    stride_scale_prior_mean: float = PF_STRIDE_SCALE_PRIOR_MEAN,
    stride_scale_init_sigma: float = PF_STRIDE_SCALE_INIT_SIGMA,
    stride_scale_retention: float = PF_STRIDE_SCALE_RETENTION,
    stride_scale_process_sigma: float = PF_STRIDE_SCALE_PROCESS_SIGMA,
    stride_scale_rejuvenation_sigma: float = PF_STRIDE_SCALE_REJUVENATION_SIGMA,
    stride_scale_min: float = PF_STRIDE_SCALE_MIN,
    stride_scale_max: float = PF_STRIDE_SCALE_MAX,
    weinberg_k: float = WEINBERG_K,
    heading_method: str = "gyro",
    step_segments: tuple[StepSegment, ...] = (),
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO,
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    motion_heading_correction: str = "auto",
    sidestep_smoothing: str = SIDESTEP_SMOOTHING_METHOD,
    forward_heading_source: str = FORWARD_HEADING_SOURCE,
    sidestep_heading_source: str = "motion",
    sidestep_suspect_mode: str = SIDESTEP_SUSPECT_MODE,
    prepared_step_headings: list[StepHeading] | None = None,
    prepared_step_lengths: list[float] | None = None,
    prepared_step_times: list[float] | None = None,
    prepared_motion_evidences: tuple[StepMotionEvidence, ...] | None = None,
    seed: int | None = None,
    heading_drift_retention: float = PF_HEADING_DRIFT_RETENTION,
    resample_ess_ratio: float = PF_RESAMPLE_ESS_RATIO,
    rejuvenation_sigma_heading: float = PF_REJUVENATION_SIGMA_HEADING,
    recovery_valid_ratio: float = PF_RECOVERY_VALID_RATIO,
    recovery_heading_sigma: float = PF_RECOVERY_HEADING_SIGMA,
    recovery_max_attempts: int = PF_RECOVERY_MAX_ATTEMPTS,
    diagnostics_collector: list[ParticleFilterStepDiagnostics] | None = None,
) -> tuple[list[list[float]], list[float], list[float], np.ndarray, list[StepHeading]]:
    """パーティクルフィルタでマップマッチング付き歩行軌跡を推定する。

    Args:
        peaks: ステップピークのインデックス配列
        df_gyro: ``low_angle`` 列を含むジャイロスコープDataFrame
        df_acc: 加速度DataFrame
        gx_mean: X軸重力成分の平均値（Y軸反転判定に使用）
        gz_mean: Z軸重力成分の平均値（Y軸反転判定に使用）
        floormap_path: フロアマップ画像のパス
        origin_px: 軌跡起点のピクセル座標
        scale: 1ピクセルあたりのメートル数
        initial_direction: 歩行開始方向のオフセット [度]
        n_particles: パーティクル数
        sigma_init_heading: 粒子ごとの初期方位ばらつき [rad]
        sigma_heading: ステップごとの方位角ノイズ [rad]
        sigma_sl_ratio: 永続倍率で説明できないステップ長ノイズの比率
        stride_scale_prior_mean: 歩幅倍率の事前中心
        stride_scale_init_sigma: 粒子ごとの初期歩幅倍率ばらつき
        stride_scale_retention: 学習した歩幅倍率偏差の保持率
        stride_scale_process_sigma: 歩幅倍率の1歩ごとの変動
        stride_scale_rejuvenation_sigma: 再標本化後の歩幅倍率多様化
        stride_scale_min: 歩幅倍率の下限
        stride_scale_max: 歩幅倍率の上限
        weinberg_k: Weinbergモデルのスケール係数
        heading_method: 方位推定手法
        step_segments: 論文寄せステップ検出の1歩区間
        sidestep_lateral_ratio: 横歩き判定に使う横方向/前方向の最小比率
        sidestep_min_lateral_displacement: 横歩き判定に必要な横方向変位の最小値
        motion_heading_correction: 水平加速度移動方向の固定ずれ補正モード
        sidestep_smoothing: 横歩き判定の平滑化モード
        forward_heading_source: forward 判定ステップの軌跡方位ソース
        seed: 乱数 seed。``None`` のときは非決定的に実行する。
        heading_drift_retention: 通常方位ドリフトを次歩へ保持する割合
        resample_ess_ratio: 適応リサンプリングを行うESS比率
        rejuvenation_sigma_heading: リサンプリング後の方位多様化ノイズ [rad]
        recovery_valid_ratio: ヒューリスティックな復旧を開始する有効粒子率
        recovery_heading_sigma: local recoveryの方位分散 [rad]
        recovery_max_attempts: recovery候補を追加生成する最大回数
        diagnostics_collector: 指定時に1歩ごとの診断値を追記するリスト

    Returns:
        tuple: (平均優先・壁際祖先フォールバック軌跡の座標リスト,
            各ステップの決定論的歩幅リスト,
            各ステップのピーク時刻リスト [s],
            全ステップのパーティクル位置 shape=(T, N, 2),
            各ステップの方位候補と採用結果)
    """
    sidestep_lateral_ratio = validate_positive_parameter(
        "sidestep_lateral_ratio",
        sidestep_lateral_ratio,
    )
    sidestep_min_lateral_displacement = validate_non_negative_parameter(
        "sidestep_min_lateral_displacement",
        sidestep_min_lateral_displacement,
    )
    selected_motion_heading_correction = validate_motion_heading_correction(
        motion_heading_correction
    )
    selected_sidestep_smoothing = validate_sidestep_smoothing(sidestep_smoothing)
    selected_forward_heading_source = validate_forward_heading_source(
        forward_heading_source
    )
    selected_sidestep_heading_source = validate_sidestep_heading_source(
        sidestep_heading_source
    )
    selected_sidestep_suspect_mode = validate_sidestep_suspect_mode(
        sidestep_suspect_mode
    )
    scale = validate_positive_parameter("scale", scale)
    if n_particles <= 0:
        raise ValueError("n_particles は正の整数を指定してください")
    sigma_sl_ratio = validate_non_negative_parameter("sigma_sl_ratio", sigma_sl_ratio)
    stride_scale_prior_mean = validate_positive_parameter(
        "stride_scale_prior_mean", stride_scale_prior_mean
    )
    stride_scale_init_sigma = validate_non_negative_parameter(
        "stride_scale_init_sigma", stride_scale_init_sigma
    )
    stride_scale_process_sigma = validate_non_negative_parameter(
        "stride_scale_process_sigma", stride_scale_process_sigma
    )
    stride_scale_retention = validate_non_negative_parameter(
        "stride_scale_retention", stride_scale_retention
    )
    if stride_scale_retention > 1.0:
        raise ValueError("stride_scale_retention は1以下を指定してください")
    stride_scale_rejuvenation_sigma = validate_non_negative_parameter(
        "stride_scale_rejuvenation_sigma", stride_scale_rejuvenation_sigma
    )
    stride_scale_min = validate_positive_parameter("stride_scale_min", stride_scale_min)
    stride_scale_max = validate_positive_parameter("stride_scale_max", stride_scale_max)
    if stride_scale_min >= stride_scale_max:
        raise ValueError("stride_scale_min は stride_scale_max 未満を指定してください")
    if not stride_scale_min <= stride_scale_prior_mean <= stride_scale_max:
        raise ValueError(
            "stride_scale_prior_mean は stride_scale_min 以上 "
            "stride_scale_max 以下を指定してください"
        )
    heading_drift_retention = validate_non_negative_parameter(
        "heading_drift_retention", heading_drift_retention
    )
    if heading_drift_retention > 1.0:
        raise ValueError("heading_drift_retention は1以下を指定してください")
    resample_ess_ratio = validate_positive_parameter(
        "resample_ess_ratio", resample_ess_ratio
    )
    if resample_ess_ratio > 1.0:
        raise ValueError("resample_ess_ratio は1以下を指定してください")
    rejuvenation_sigma_heading = validate_non_negative_parameter(
        "rejuvenation_sigma_heading", rejuvenation_sigma_heading
    )
    recovery_valid_ratio = validate_non_negative_parameter(
        "recovery_valid_ratio", recovery_valid_ratio
    )
    if recovery_valid_ratio > 1.0:
        raise ValueError("recovery_valid_ratio は1以下を指定してください")
    recovery_heading_sigma = validate_positive_parameter(
        "recovery_heading_sigma", recovery_heading_sigma
    )
    if recovery_max_attempts <= 0:
        raise ValueError("recovery_max_attempts は正の整数を指定してください")
    rng = np.random.default_rng(seed)
    stride_rng = np.random.default_rng(
        None if seed is None else np.random.SeedSequence([seed, 0x53545249])
    )
    motion_rng = np.random.default_rng(
        None if seed is None else np.random.SeedSequence([seed, 0x4D4F544E])
    )

    # フロアマップをグレースケールで読み込み
    map_gray = _normalize_floormap_gray(plt.imread(Path(floormap_path)))
    if map_gray.ndim != 2 or map_gray.size == 0:
        raise ValueError("フロアマップは空でない2次元画像を指定してください")

    # 全パーティクルを原点で初期化（[x, y] の2次元状態）
    particles = np.zeros((n_particles, 2))
    if not bool(
        _evaluate_particle_transitions(
            particles[:1],
            particles[:1],
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )[0]
    ):
        raise ValueError("origin_px は歩行可能なマップ内画素を指定してください")
    heading_correction = np.zeros(n_particles, dtype=float)
    heading_drift = rng.normal(0, sigma_init_heading, n_particles)
    motion_state = np.full(n_particles, _MOTION_FORWARD, dtype=np.int8)
    stride_scale = np.clip(
        stride_scale_prior_mean
        + stride_rng.normal(0.0, stride_scale_init_sigma, n_particles),
        stride_scale_min,
        stride_scale_max,
    )
    weights = np.ones(n_particles) / n_particles

    step_lengths: list[float] = []
    t_at_steps: list[float] = []
    position_history: list[np.ndarray] = [particles.copy()]
    heading_correction_history: list[np.ndarray] = [heading_correction.copy()]
    heading_drift_history: list[np.ndarray] = [heading_drift.copy()]
    motion_state_history: list[np.ndarray] = [motion_state.copy()]
    stride_scale_history: list[np.ndarray] = [stride_scale.copy()]
    weight_history: list[np.ndarray] = [weights.copy()]
    parent_history: list[np.ndarray] = []
    all_particles_list: list[np.ndarray] = [particles.copy()]  # ステップ0（原点）
    step_headings: list[StepHeading] = []
    healthy_checkpoint_steps = [0]
    diagnostics_start_index = (
        len(diagnostics_collector) if diagnostics_collector is not None else 0
    )
    using_prepared_steps = (
        prepared_step_headings is not None
        and prepared_step_lengths is not None
        and prepared_step_times is not None
    )
    if not using_prepared_steps and (
        prepared_step_headings is not None
        or prepared_step_lengths is not None
        or prepared_step_times is not None
    ):
        raise ValueError(
            "prepared_step_headings, prepared_step_lengths, "
            "prepared_step_times はすべて同時に指定してください"
        )
    if prepared_motion_evidences is not None and not using_prepared_steps:
        raise ValueError(
            "prepared_motion_evidences は prepared step 一式と同時に指定してください"
        )
    if not using_prepared_steps:
        device_orientation_mode = estimate_device_orientation_mode(
            df_acc,
            df_gyro,
            peaks,
            initial_direction,
            step_segments=step_segments,
        )
        motion_heading_correction_rad = resolve_motion_heading_correction(
            df_acc,
            df_gyro,
            peaks,
            initial_direction,
            step_segments,
            selected_motion_heading_correction,
            device_orientation_mode,
        )

        phi_0 = (
            estimate_initial_forward_angle(df_acc, df_gyro, peaks)
            if STEP_LENGTH_METHOD == "forward"
            else 0.0
        )
        raw_step_headings: list[StepHeading] = []
        raw_step_lengths: list[float] = []
        raw_step_times: list[float] = []
        for i, p in enumerate(peaks):
            if p >= len(df_acc):
                continue
            if STEP_LENGTH_METHOD == "forward" and i + 1 >= len(peaks):
                continue

            step_heading = resolve_step_heading(
                peaks,
                df_gyro,
                df_acc,
                i,
                initial_direction=initial_direction,
                heading_method=heading_method,
                step_segments=step_segments,
                motion_heading_correction=motion_heading_correction_rad,
                sidestep_lateral_ratio=sidestep_lateral_ratio,
                sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
                device_orientation_mode=device_orientation_mode,
            )
            if step_heading.selected_heading is None:
                continue

            if STEP_LENGTH_METHOD == "forward":
                sl_det = estimate_step_length_forward(df_acc, df_gyro, peaks, i, phi_0)
            else:
                sl_det = estimate_step_length(df_acc, int(p), k=weinberg_k)
            raw_step_headings.append(step_heading)
            raw_step_lengths.append(sl_det)
            raw_step_times.append(
                step_output_time(df_acc, peaks, i, STEP_LENGTH_METHOD)
            )

        smoothed_step_headings = smooth_step_headings(
            raw_step_headings,
            selected_sidestep_smoothing,
            selected_sidestep_suspect_mode,
        )
        stabilized_step_headings = stabilize_trajectory_headings(
            smoothed_step_headings,
            selected_forward_heading_source,
            selected_sidestep_heading_source,
        )
        motion_evidences = build_step_motion_evidences(stabilized_step_headings)
    else:
        assert prepared_step_headings is not None
        assert prepared_step_lengths is not None
        assert prepared_step_times is not None
        if not (
            len(prepared_step_headings)
            == len(prepared_step_lengths)
            == len(prepared_step_times)
        ):
            raise ValueError(
                "prepared_step_headings, prepared_step_lengths, "
                "prepared_step_times の長さが一致しません"
            )
        stabilized_step_headings = prepared_step_headings
        raw_step_lengths = prepared_step_lengths
        raw_step_times = prepared_step_times
        motion_evidences = (
            build_step_motion_evidences(stabilized_step_headings)
            if prepared_motion_evidences is None
            else prepared_motion_evidences
        )
        if len(motion_evidences) != len(stabilized_step_headings):
            raise ValueError(
                "prepared_motion_evidences と prepared_step_headings の"
                "長さが一致しません"
            )
    particle_motion_headings = build_particle_motion_headings(stabilized_step_headings)

    previous_heading: float | None = None

    for step_number, (
        step_heading,
        sl_det,
        step_time,
        motion_evidence,
        particle_heading,
    ) in enumerate(
        zip(
            stabilized_step_headings,
            raw_step_lengths,
            raw_step_times,
            motion_evidences,
            particle_motion_headings,
            strict=True,
        ),
        start=1,
    ):
        if particle_heading is None:
            continue
        angle_det = particle_heading
        step_heading = step_heading._replace(
            selected_heading=particle_heading,
            source="particle_evidence_motion",
        )
        if using_prepared_steps:
            pass
        else:
            step_motion = estimate_step_motion(
                step_heading,
                sl_det,
                previous_heading,
                selected_forward_heading_source,
                selected_sidestep_heading_source,
                selected_sidestep_suspect_mode,
            )
            if step_motion is None:
                continue
            angle_det = step_motion.heading
            sl_det = step_motion.length
            step_heading = step_heading._replace(
                selected_heading=step_motion.heading,
                source=step_heading.source
                if step_heading.source.startswith("trajectory_")
                else "state_motion",
                step_length_scale=step_motion.length_scale,
                trajectory_movement_type=step_motion.movement_type,
                forward_heading_source=selected_forward_heading_source,
            )

        particles_before = particles.copy()
        heading_correction_before = heading_correction.copy()
        heading_drift_before = heading_drift.copy()
        stride_scale_before = stride_scale.copy()
        motion_state_before = motion_state.copy()
        weights_before = weights.copy()
        ess_before_observation = _effective_sample_size(weights_before)

        # 通常ドリフトだけを平均回帰させ、recovery補正は独立に保持する。
        proposed_correction = heading_correction_before
        proposed_drift = _normalize_angle(
            heading_drift_retention * heading_drift_before
            + rng.normal(0, sigma_heading, n_particles)
        )
        observation_likelihoods = _motion_state_likelihoods(motion_evidence)
        proposed_motion_state, _state_predictive_likelihoods = _sample_motion_states(
            motion_state_before,
            observation_likelihoods,
            motion_rng,
        )
        state_headings = _motion_state_headings(
            step_heading,
            motion_evidence,
            particle_heading,
        )
        particle_base_headings = state_headings[proposed_motion_state]
        theta = particle_base_headings + proposed_correction + proposed_drift
        proposed_stride_scale = np.clip(
            stride_scale_prior_mean
            + stride_scale_retention * (stride_scale_before - stride_scale_prior_mean)
            + stride_rng.normal(0.0, stride_scale_process_sigma, n_particles),
            stride_scale_min,
            stride_scale_max,
        )
        if (
            motion_evidence.calibration_reliability >= 0.85
            or step_heading.sidestep_cluster_id is None
        ):
            particle_step_lengths = np.full(n_particles, sl_det, dtype=float)
        else:
            raw_step_length = sl_det / max(step_heading.step_length_scale, 1e-12)
            state_length_scales = np.asarray(
                [
                    1.0,
                    SIDESTEP_LENGTH_SCALE,
                    SIDESTEP_LENGTH_SCALE,
                    TURNING_LENGTH_SCALE,
                ]
            )
            particle_step_lengths = (
                raw_step_length * state_length_scales[proposed_motion_state]
            )
        sl = np.clip(
            particle_step_lengths
            * proposed_stride_scale
            * (1 + rng.normal(0, sigma_sl_ratio, n_particles)),
            0,
            None,
        )
        proposed_particles = particles_before.copy()
        proposed_particles[:, 0] += sl * np.cos(theta)
        proposed_particles[:, 1] += sl * np.sin(theta)

        valid_transition = _evaluate_particle_transitions(
            particles_before,
            proposed_particles,
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        valid_count = int(np.count_nonzero(valid_transition))
        valid_weight_mask = valid_transition & (weights_before > 0.0)
        valid_weight_count = int(np.count_nonzero(valid_weight_mask))
        posterior_weights = weights_before * valid_transition.astype(float)
        valid_weight_mass = float(posterior_weights.sum())
        ess_after_observation = (
            _effective_sample_size(posterior_weights / valid_weight_mass)
            if valid_weight_mass > 0.0
            else 0.0
        )
        recovery_attempted = (
            valid_weight_mass <= 0.0
            or valid_weight_count / n_particles < recovery_valid_ratio
        )
        recovery_mode = "none"
        recovery_valid_count = 0
        recovery_attempts = 0
        recovery_heading_delta_deg: float | None = None
        recovery_step_scale: float | None = None
        recovery_cost: float | None = None
        recovery_checkpoint_step: int | None = None
        recovery_replay_steps = 0
        resampled = False

        if recovery_attempted:
            recovery = _generate_recovery_candidates(
                particles_before,
                heading_correction_before,
                heading_drift_before,
                stride_scale_before,
                proposed_motion_state,
                weights_before,
                particle_base_headings,
                particle_step_lengths,
                sigma_sl_ratio,
                n_particles,
                map_gray,
                gx_mean,
                gz_mean,
                origin_px,
                scale,
                recovery_heading_sigma,
                recovery_max_attempts,
                rng,
                allow_turn_candidates=(
                    step_heading.trajectory_movement_type == "turning"
                    or step_heading.movement_type == "turning"
                    or "sidestep"
                    in (
                        step_heading.trajectory_movement_type
                        or step_heading.movement_type
                    )
                    or (
                        bool(step_headings)
                        and (
                            step_headings[-1].trajectory_movement_type == "turning"
                            or step_headings[-1].movement_type == "turning"
                        )
                    )
                ),
            )
            if recovery is None:
                completed_steps = len(step_lengths)
                checkpoint_step = next(
                    (
                        candidate
                        for candidate in healthy_checkpoint_steps
                        if 0 < completed_steps - candidate <= 3
                    ),
                    None,
                )
                replay_result = None
                if checkpoint_step is not None:
                    replay_headings = step_headings[checkpoint_step:] + [step_heading]
                    replay_angles = np.asarray(
                        [heading.selected_heading for heading in replay_headings],
                        dtype=float,
                    )
                    replay_lengths = np.asarray(
                        step_lengths[checkpoint_step:] + [sl_det],
                        dtype=float,
                    )
                    replay_result = _replay_from_checkpoint(
                        position_history[checkpoint_step],
                        heading_correction_history[checkpoint_step],
                        heading_drift_history[checkpoint_step],
                        stride_scale_history[checkpoint_step],
                        weight_history[checkpoint_step],
                        replay_angles,
                        replay_lengths,
                        n_particles,
                        map_gray,
                        gx_mean,
                        gz_mean,
                        origin_px,
                        scale,
                        recovery_heading_sigma,
                        rng,
                        checkpoint_motion_state=motion_state_history[checkpoint_step],
                    )
                if replay_result is None:
                    fallback_recovery = _generate_recovery_candidates(
                        particles_before,
                        heading_correction_before,
                        heading_drift_before,
                        stride_scale_before,
                        proposed_motion_state,
                        weights_before,
                        particle_base_headings,
                        particle_step_lengths,
                        sigma_sl_ratio,
                        n_particles,
                        map_gray,
                        gx_mean,
                        gz_mean,
                        origin_px,
                        scale,
                        recovery_heading_sigma,
                        recovery_max_attempts,
                        rng,
                        allow_turn_candidates=True,
                    )
                    if fallback_recovery is None:
                        # 全候補とreplayが失敗した場合だけ直前位置を保持する。
                        particles = particles_before
                        heading_correction = heading_correction_before
                        heading_drift = heading_drift_before
                        stride_scale = stride_scale_before
                        motion_state = proposed_motion_state
                        weights = weights_before
                        parent_indices = np.arange(n_particles, dtype=int)
                        recovery_mode = "failed_hold"
                        recovery_attempts = recovery_max_attempts
                    else:
                        particles = fallback_recovery.particles
                        heading_correction = fallback_recovery.heading_correction
                        heading_drift = fallback_recovery.heading_drift
                        stride_scale = fallback_recovery.stride_scale
                        motion_state = fallback_recovery.motion_state
                        weights = np.full(n_particles, 1.0 / n_particles)
                        parent_indices = fallback_recovery.parent_indices
                        recovery_mode = f"fallback_{fallback_recovery.mode}"
                        recovery_valid_count = fallback_recovery.valid_count
                        recovery_attempts = fallback_recovery.attempts
                        recovery_heading_delta_deg = fallback_recovery.heading_delta_deg
                        recovery_step_scale = fallback_recovery.step_scale
                        recovery_cost = fallback_recovery.mean_cost
                        resampled = True
                else:
                    recovery = replay_result.recovery
                    recovery = replace(
                        recovery,
                        motion_state=proposed_motion_state.copy(),
                    )
                    assert checkpoint_step is not None
                    replay_depth = replay_result.replay_positions.shape[0]
                    position_history = position_history[: checkpoint_step + 1]
                    position_history.extend(replay_result.replay_positions[:-1].copy())
                    all_particles_list = all_particles_list[: checkpoint_step + 1]
                    all_particles_list.extend(
                        replay_result.replay_positions[:-1].copy()
                    )
                    parent_history = parent_history[:checkpoint_step]
                    if replay_depth > 1:
                        parent_history.append(recovery.parent_indices.copy())
                        parent_history.extend(
                            np.arange(n_particles, dtype=int)
                            for _ in range(replay_depth - 2)
                        )
                    heading_correction_history = heading_correction_history[
                        : checkpoint_step + 1
                    ]
                    heading_drift_history = heading_drift_history[: checkpoint_step + 1]
                    motion_state_history = motion_state_history[: checkpoint_step + 1]
                    stride_scale_history = stride_scale_history[: checkpoint_step + 1]
                    weight_history = weight_history[: checkpoint_step + 1]
                    healthy_checkpoint_steps = [
                        step
                        for step in healthy_checkpoint_steps
                        if step <= checkpoint_step
                    ]
                    for _ in range(replay_depth - 1):
                        heading_correction_history.append(
                            recovery.heading_correction.copy()
                        )
                        heading_drift_history.append(recovery.heading_drift.copy())
                        motion_state_history.append(recovery.motion_state.copy())
                        stride_scale_history.append(recovery.stride_scale.copy())
                        weight_history.append(np.full(n_particles, 1.0 / n_particles))
                    particles = recovery.particles
                    heading_correction = recovery.heading_correction
                    heading_drift = recovery.heading_drift
                    stride_scale = recovery.stride_scale
                    motion_state = recovery.motion_state
                    weights = np.full(n_particles, 1.0 / n_particles)
                    parent_indices = (
                        np.arange(n_particles, dtype=int)
                        if replay_depth > 1
                        else recovery.parent_indices
                    )
                    recovery_mode = recovery.mode
                    recovery_valid_count = recovery.valid_count
                    recovery_attempts = recovery.attempts
                    recovery_heading_delta_deg = recovery.heading_delta_deg
                    recovery_step_scale = recovery.step_scale
                    recovery_cost = recovery.mean_cost
                    recovery_checkpoint_step = checkpoint_step
                    recovery_replay_steps = replay_depth
                    resampled = True
            else:
                particles = recovery.particles
                heading_correction = recovery.heading_correction
                heading_drift = recovery.heading_drift
                stride_scale = recovery.stride_scale
                motion_state = recovery.motion_state
                weights = np.full(n_particles, 1.0 / n_particles)
                parent_indices = recovery.parent_indices
                recovery_mode = recovery.mode
                recovery_valid_count = recovery.valid_count
                recovery_attempts = recovery.attempts
                recovery_heading_delta_deg = recovery.heading_delta_deg
                recovery_step_scale = recovery.step_scale
                recovery_cost = recovery.mean_cost
                resampled = True
        else:
            if valid_weight_mass <= 0.0:
                raise RuntimeError("内部エラー: recoveryせず粒子重みが全滅しました")
            posterior_weights /= valid_weight_mass
            if ess_after_observation < resample_ess_ratio * n_particles:
                indices = _systematic_resample(posterior_weights, rng)
                particles = proposed_particles[indices]
                heading_correction = proposed_correction[indices]
                heading_drift = proposed_drift[indices]
                stride_scale = proposed_stride_scale[indices]
                motion_state = proposed_motion_state[indices]
                if rejuvenation_sigma_heading > 0.0:
                    heading_drift = _normalize_angle(
                        heading_drift
                        + rng.normal(0, rejuvenation_sigma_heading, n_particles)
                    )
                if stride_scale_rejuvenation_sigma > 0.0:
                    stride_scale = np.clip(
                        stride_scale
                        + stride_rng.normal(
                            0.0,
                            stride_scale_rejuvenation_sigma,
                            n_particles,
                        ),
                        stride_scale_min,
                        stride_scale_max,
                    )
                weights = np.full(n_particles, 1.0 / n_particles)
                parent_indices = indices
                resampled = True
            else:
                particles = proposed_particles
                heading_correction = proposed_correction
                heading_drift = proposed_drift
                stride_scale = proposed_stride_scale
                motion_state = proposed_motion_state
                weights = posterior_weights
                parent_indices = np.arange(n_particles, dtype=int)

        ess_after_resampling = _effective_sample_size(weights)

        position_history.append(particles.copy())
        heading_correction_history.append(heading_correction.copy())
        heading_drift_history.append(heading_drift.copy())
        motion_state_history.append(motion_state.copy())
        stride_scale_history.append(stride_scale.copy())
        weight_history.append(weights.copy())
        step_lengths.append(sl_det)
        t_at_steps.append(step_time)
        step_headings.append(step_heading)
        previous_heading = angle_det

        parent_history.append(parent_indices)
        all_particles_list.append(particles.copy())
        if (recovery_mode == "none" and valid_weight_count > 0) or (
            recovery_mode not in {"none", "failed_hold"} and recovery_valid_count > 0
        ):
            healthy_checkpoint_steps.append(len(step_lengths))
        if diagnostics_collector is not None:
            unique_position_count = int(
                np.unique(np.round(particles, decimals=9), axis=0).shape[0]
            )
            position_center = np.average(particles, axis=0, weights=weights)
            position_spread_rms_m = float(
                np.sqrt(
                    np.sum(
                        weights * np.sum(np.square(particles - position_center), axis=1)
                    )
                )
            )
            heading_drift_std_deg = float(
                np.degrees(_weighted_circular_std(heading_drift, weights))
            )
            heading_total_std_deg = float(
                np.degrees(
                    _weighted_circular_std(
                        _normalize_angle(heading_correction + heading_drift),
                        weights,
                    )
                )
            )
            stride_scale_mean = float(np.sum(weights * stride_scale))
            stride_scale_std = float(
                np.sqrt(np.sum(weights * np.square(stride_scale - stride_scale_mean)))
            )
            effective_step_lengths = (
                particle_step_lengths[parent_indices] * stride_scale
                if resampled
                else particle_step_lengths * stride_scale
            )
            effective_step_length_mean_m = float(
                np.sum(weights * effective_step_lengths)
            )
            effective_step_length_std_m = float(
                np.sqrt(
                    np.sum(
                        weights
                        * np.square(
                            effective_step_lengths - effective_step_length_mean_m
                        )
                    )
                )
            )
            state_probabilities = np.asarray(
                [float(np.sum(weights[motion_state == state])) for state in range(4)]
            )
            representative_state_index = int(np.argmax(state_probabilities))
            positive_state_probabilities = state_probabilities[
                state_probabilities > 0.0
            ]
            motion_state_entropy = float(
                -np.sum(
                    positive_state_probabilities * np.log(positive_state_probabilities)
                )
            )
            parent_states = motion_state_before[parent_indices]
            motion_state_transition_count = int(
                np.count_nonzero(motion_state != parent_states)
            )
            diagnostics_collector.append(
                ParticleFilterStepDiagnostics(
                    step=step_number,
                    timestamp_s=float(step_time),
                    valid_count=valid_count,
                    valid_ratio=valid_count / n_particles,
                    valid_weight_count=valid_weight_count,
                    valid_weight_mass_before_normalization=valid_weight_mass,
                    ess_before_observation=ess_before_observation,
                    ess_after_observation=ess_after_observation,
                    ess_after_resampling=ess_after_resampling,
                    max_weight=float(np.max(weights)),
                    unique_parent_count=int(np.unique(parent_indices).size),
                    unique_position_count=unique_position_count,
                    position_spread_rms_m=position_spread_rms_m,
                    heading_drift_std_deg=heading_drift_std_deg,
                    heading_total_std_deg=heading_total_std_deg,
                    stride_scale_mean=stride_scale_mean,
                    stride_scale_std=stride_scale_std,
                    effective_step_length_mean_m=effective_step_length_mean_m,
                    effective_step_length_std_m=effective_step_length_std_m,
                    forward_state_probability=float(state_probabilities[0]),
                    sidestep_left_state_probability=float(state_probabilities[1]),
                    sidestep_right_state_probability=float(state_probabilities[2]),
                    turning_state_probability=float(state_probabilities[3]),
                    representative_motion_state=_MOTION_STATE_NAMES[
                        representative_state_index
                    ],
                    motion_state_entropy=motion_state_entropy,
                    motion_state_transition_count=motion_state_transition_count,
                    motion_reliability=motion_evidence.motion_reliability,
                    calibration_reliability=(motion_evidence.calibration_reliability),
                    resampled=resampled,
                    recovery_attempted=recovery_attempted,
                    recovery_mode=recovery_mode,
                    recovery_valid_count=recovery_valid_count,
                    recovery_attempts=recovery_attempts,
                    recovery_heading_delta_deg=recovery_heading_delta_deg,
                    recovery_step_scale=recovery_step_scale,
                    recovery_cost=recovery_cost,
                    recovery_checkpoint_step=recovery_checkpoint_step,
                    recovery_replay_steps=recovery_replay_steps,
                    trajectory_mode="pending",
                    trajectory_source_index=None,
                )
            )

    all_particles = np.stack(all_particles_list)  # shape: (T+1, N, 2)
    particle_paths = _reconstruct_particle_paths(position_history, parent_history)
    selected_path, trajectory_modes, trajectory_sources = _select_reachable_mean_path(
        particle_paths,
        weights,
        map_gray,
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    if diagnostics_collector is not None:
        for diagnostic_offset, (mode, source) in enumerate(
            zip(trajectory_modes[1:], trajectory_sources[1:], strict=True)
        ):
            collector_index = diagnostics_start_index + diagnostic_offset
            diagnostics_collector[collector_index] = replace(
                diagnostics_collector[collector_index],
                trajectory_mode=mode,
                trajectory_source_index=source,
            )
    return (
        selected_path.tolist(),
        step_lengths,
        t_at_steps,
        all_particles,
        step_headings,
    )


def plot_particle_filter_trajectory(
    trajectory: list[list[float]],
    gx_mean: float = 0.0,
    gz_mean: float = 0.0,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    output_dir: Path | None = None,
    step_headings: list[StepHeading] | None = None,
) -> None:
    """PF の平均優先・壁際祖先フォールバック軌跡を描画する。

    Args:
        trajectory: 各ステップの [x, y] 座標リスト（メートル）
        gx_mean: X軸重力成分の平均値
        gz_mean: Z軸重力成分の平均値
        floormap_path: フロアマップ画像のパス
        origin_px: 軌跡起点のピクセル座標
        scale: 1ピクセルあたりのメートル数
        output_dir: 出力ディレクトリ（指定時に PNG 保存）
    """
    df = pd.DataFrame(trajectory, columns=["x", "y"])
    px, py = compute_pixel_coords(
        df["x"].to_numpy(), df["y"].to_numpy(), gx_mean, gz_mean, origin_px, scale
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    map_img = plt.imread(Path(floormap_path))
    ax.imshow(map_img)

    n = len(px)
    norm = Normalize(vmin=0, vmax=max(n - 1, 1))
    cmap = cm.get_cmap("plasma")
    pts = np.column_stack([px, py]).reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segments.tolist(), cmap=cmap, norm=norm, zorder=2)
    lc.set_array(np.arange(n - 1))
    ax.add_collection(lc)
    sc = ax.scatter(px, py, c=np.arange(n), cmap=cmap, norm=norm, s=20, zorder=3)
    fig.colorbar(sc, ax=ax, label="Step")
    ax.plot(px[0], py[0], "go", markersize=10, label="Start", zorder=4)
    plot_heading_overlay(
        ax,
        trajectory,
        step_headings,
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )

    ax.set_title("Particle Filter Trajectory on Floormap")
    ax.legend()
    plt.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / "pf_trajectory.png", dpi=150, bbox_inches="tight")
    plt.show()


def save_particle_animation(
    all_particles: np.ndarray,
    mean_trajectory: list[list[float]],
    gx_mean: float,
    gz_mean: float,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    output_path: Path | str = Path("output/particle_filter.mp4"),
    fps: int = 10,
) -> None:
    """PF の各ステップのパーティクル分布をフロアマップ上に描画し MP4 として保存する。

    Args:
        all_particles: 全ステップのパーティクル位置 shape=(T, N, 2)
        mean_trajectory: 平均優先・壁際祖先フォールバック軌跡
            （互換性のため既存引数名を維持）
        gx_mean: X軸重力成分の平均値
        gz_mean: Z軸重力成分の平均値
        floormap_path: フロアマップ画像のパス
        origin_px: 軌跡起点のピクセル座標
        scale: 1ピクセルあたりのメートル数
        output_path: 出力ファイルパス（.mp4）
        fps: フレームレート
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: PLC0415

    map_img = plt.imread(Path(floormap_path))
    representative_arr = np.array(mean_trajectory)  # shape: (T, 2)

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame: int) -> list[Artist]:
        ax.cla()
        ax.imshow(map_img)

        # 全パーティクルを半透明グレーで描画
        px_p, py_p = compute_pixel_coords(
            all_particles[frame, :, 0],
            all_particles[frame, :, 1],
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        ax.scatter(px_p, py_p, s=30, c="cyan", alpha=0.5, zorder=2)

        # ステップ 0 〜 現在の選択軌跡を青線で描画
        if frame > 0:
            px_m, py_m = compute_pixel_coords(
                representative_arr[: frame + 1, 0],
                representative_arr[: frame + 1, 1],
                gx_mean,
                gz_mean,
                origin_px,
                scale,
            )
            ax.plot(px_m, py_m, "b-", linewidth=1.5, zorder=3)

        # 現ステップの選択位置を赤点で描画
        px_c, py_c = compute_pixel_coords(
            representative_arr[frame : frame + 1, 0],
            representative_arr[frame : frame + 1, 1],
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        ax.scatter(px_c, py_c, s=60, c="red", zorder=4)
        ax.set_title(f"Step {frame}")
        return []

    anim = FuncAnimation(fig, update, frames=len(all_particles), interval=1000 // fps)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(output_path), writer=FFMpegWriter(fps=fps))
        print(f"Animation saved to {output_path}")
    except Exception:
        gif_path = output_path.with_suffix(".gif")
        anim.save(str(gif_path), writer=PillowWriter(fps=fps))
        print(f"ffmpeg が見つかりません。GIF として保存しました: {gif_path}")
    plt.close(fig)
