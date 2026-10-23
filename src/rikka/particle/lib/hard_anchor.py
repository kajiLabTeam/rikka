"""PFの確定位置における前向き区間の再開始。

役割: 根拠付き位置を厳密に適用し、過去の軌跡を固定した独立区間を開始する。
依存元: commonの座標変換と観測型、粒子の経路選択と歩行可能性判定を使用する。
利用先: initializeが入力を検証し、runnerが地図制約解決後に位置を適用する。
処理フロー: 歩境界との一致と地図を検証し、補正前区間を選択して保存した後、
    全粒子の位置だけを確定座標へ移し、状態仮説を保ったまま履歴探索を区切る。
"""

import numpy as np

from ...common.lib.floormap import compute_meter_coords
from ...common.lib.models import HardAnchorCorrection
from .landmark import meter_walkable_mask
from .path_history import _reconstruct_particle_paths
from .path_selection import _select_reachable_cluster_path
from .sequence_path import _select_sequence_map_path, _unsupported_reversal_count
from .state import ParticleRuntime


def initialize_hard_positions(ctx: ParticleRuntime) -> None:
    """歩間を最近傍歩へ丸めず、曖昧な時刻・重複・歩行不可座標を拒否する。"""
    ids: set[str] = set()
    for observation in ctx.hard_positions:
        if (
            observation.floor_id is not None
            and observation.floor_id != ctx.floormap_floor_id
        ):
            raise ValueError("確定位置のfloor_idと使用地図のfloor_idを照合できません")
        if observation.event_id in ids:
            raise ValueError("確定位置の event_id が重複しています")
        ids.add(observation.event_id)
        matches = np.flatnonzero(
            np.isclose(ctx.raw_step_times, observation.timestamp_s, rtol=0, atol=1e-9)
        )
        if len(matches) != 1:
            raise ValueError(
                "確定位置の時刻は1つの歩境界と一致させてください（歩間補間は未対応）"
            )
        step = int(matches[0]) + 1
        if step in ctx.hard_anchor_by_step:
            raise ValueError("同じ歩への確定位置の重複は指定できません")
        if any(item is None for item in ctx.particle_motion_headings[:step]):
            raise ValueError("方位未定義で省略される歩には確定位置を指定できません")
        xy = hard_position_meters(ctx, observation.pixel_x, observation.pixel_y)
        if not meter_walkable_mask(
            xy[None, :],
            ctx.map_gray,
            ctx.gx_mean,
            ctx.gz_mean,
            ctx.origin_px,
            ctx.scale,
        )[0]:
            raise ValueError("確定位置は地図内の歩行可能座標を指定してください")
        ctx.hard_anchor_by_step[step] = observation
    ctx.hard_anchor_events = (
        []
        if ctx.hard_anchor_events_collector is None
        else ctx.hard_anchor_events_collector
    )


def hard_position_meters(
    ctx: ParticleRuntime, pixel_x: float, pixel_y: float
) -> np.ndarray:
    """共通の地図変換を使い、確定位置を開始点基準メートルへ変換する。"""
    x, y = compute_meter_coords(
        np.asarray([pixel_x]),
        np.asarray([pixel_y]),
        ctx.gx_mean,
        ctx.gz_mean,
        ctx.origin_px,
        ctx.scale,
    )
    return np.asarray([x[0], y[0]], dtype=float)


def select_hard_segment(
    ctx: ParticleRuntime, *, pending: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """直前の確定位置から現在までのcurrentと単一祖先経路を独立に選ぶ。"""
    start = ctx.last_hard_anchor_step
    positions = ctx.position_history[start:]
    weights = ctx.weight_history[start:]
    parents = ctx.parent_history[start:]
    headings = list(ctx.step_headings[start:])
    if pending:
        positions = [*positions, ctx.particles.copy()]
        weights = [*weights, ctx.weights.copy()]
        parents = [*parents, ctx.parent_indices.copy()]
        headings.append(ctx.step_heading)
    current, _, _ = _select_reachable_cluster_path(
        positions,
        weights,
        parents,
        ctx.map_gray,
        ctx.gx_mean,
        ctx.gz_mean,
        ctx.origin_px,
        ctx.scale,
    )
    particle_paths = _reconstruct_particle_paths(positions, parents)
    sensor = np.asarray([item.selected_heading for item in headings], dtype=float)
    turning = np.asarray(
        [
            item.trajectory_movement_type == "turning"
            or item.movement_type == "turning"
            for item in headings
        ]
    )
    scores = ctx.next_path_log_scores if pending else ctx.path_log_score_history[-1]
    sequence, _, _ = _select_sequence_map_path(
        particle_paths,
        scores,
        ctx.map_gray,
        ctx.gx_mean,
        ctx.gz_mean,
        ctx.origin_px,
        ctx.scale,
        sensor,
        turning,
        None,
    )
    return current, sequence


def hard_segment_reversals(
    ctx: ParticleRuntime, path: np.ndarray, *, pending: bool = False
) -> int:
    """補正そのものを歩行反転に含めず、区間内の反転だけを数える。"""
    headings = list(ctx.step_headings[ctx.last_hard_anchor_step :])
    if pending:
        headings.append(ctx.step_heading)
    sensor = np.asarray([item.selected_heading for item in headings], dtype=float)
    turning = np.asarray(
        [
            item.trajectory_movement_type == "turning"
            or item.movement_type == "turning"
            for item in headings
        ]
    )
    return _unsupported_reversal_count(path, sensor, turning)


def apply_hard_position(ctx: ParticleRuntime) -> None:
    """位置だけを厳密に訂正し、位置から方位・歩幅を逆算しない。"""
    observation = ctx.hard_anchor_by_step.get(ctx.step_number)
    if observation is None:
        return
    current, sequence = select_hard_segment(ctx, pending=True)
    ctx.hard_current_reversals += hard_segment_reversals(ctx, current, pending=True)
    ctx.hard_sequence_reversals += hard_segment_reversals(ctx, sequence, pending=True)
    target = hard_position_meters(ctx, observation.pixel_x, observation.pixel_y)
    before = (sequence[-1] if ctx.path_selection == "sequence" else current[-1]).copy()
    current[-1] = target
    sequence[-1] = target
    ctx.hard_current_prefix.extend(
        current.tolist()[1:] if ctx.hard_current_prefix else current.tolist()
    )
    ctx.hard_sequence_prefix.extend(
        sequence.tolist()[1:] if ctx.hard_sequence_prefix else sequence.tolist()
    )
    ctx.particles[:] = target
    if observation.heading_deg is not None:
        desired = np.radians(observation.heading_deg)
        ctx.heading_correction[:] = np.arctan2(
            np.sin(desired - ctx.particle_base_headings),
            np.cos(desired - ctx.particle_base_headings),
        )
        ctx.heading_drift[:] = 0.0
    ctx.next_path_log_scores[:] = 0.0
    ctx.last_hard_anchor_step = ctx.step_number
    ctx.healthy_checkpoint_steps = [ctx.step_number]
    ctx.hard_anchor_events.append(
        HardAnchorCorrection(
            ctx.step_number,
            observation.event_id,
            observation.timestamp_s,
            float(before[0]),
            float(before[1]),
            float(target[0]),
            float(target[1]),
            observation.source,
            observation.evidence,
        )
    )
