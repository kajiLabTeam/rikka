"""particle filter の代表軌跡確定。

役割:
    代表軌跡確定を独立した段階として実装する。
依存元:
    common の共有型・設定と particle/lib の部品、ParticleRuntime を利用する。
利用先:
    particle/lib/runner が元の実行順序どおりに呼び出す。
処理フロー:
    履歴から代表軌跡を選択し、互換戻り値を組み立てる。
"""

from dataclasses import replace

import numpy as np

from ...common.lib.models import StepHeading
from ...landmark.lib.retrofit import (
    apply_transform,
    count_walkability_violations,
    solve_anchor_similarity,
)
from ...particle.lib.recorder import (
    ParticlePathComparison,
)
from .path_history import _reconstruct_particle_paths
from .path_selection import _select_reachable_cluster_path
from .sequence_path import (
    _select_sequence_map_path,
    _unsupported_reversal_count,
)
from .state import ParticleRuntime


def _retrofit_landmark_anchor_jumps(ctx: ParticleRuntime) -> None:
    """代表経路の確定アンカー直前区間を相似変換し、位置ジャンプを除く。"""
    if not ctx.landmark_retrofit or not ctx.landmark_anchor_steps:
        return
    retrofitted = ctx.selected_path.tolist()
    previous_anchor = 0
    for anchor_step in sorted(ctx.landmark_anchor_steps):
        raw_endpoint_index = anchor_step - 1
        if raw_endpoint_index <= previous_anchor:
            previous_anchor = anchor_step
            continue
        transform = solve_anchor_similarity(
            tuple(retrofitted[previous_anchor]),
            tuple(retrofitted[raw_endpoint_index]),
            tuple(retrofitted[anchor_step]),
        )
        if transform is None:
            previous_anchor = anchor_step
            continue
        candidate = apply_transform(
            retrofitted,
            transform,
            previous_anchor + 1,
            raw_endpoint_index,
        )
        if (
            count_walkability_violations(
                candidate,
                map_gray=ctx.map_gray,
                gx_mean=ctx.gx_mean,
                gz_mean=ctx.gz_mean,
                origin_px=ctx.origin_px,
                scale=ctx.scale,
            )
            == 0
        ):
            retrofitted = candidate
        previous_anchor = anchor_step
    ctx.selected_path = np.asarray(retrofitted, dtype=float)


def finalize(
    ctx: ParticleRuntime,
) -> tuple[
    list[list[float]],
    list[float],
    list[float],
    np.ndarray,
    list[StepHeading],
]:
    allowed_jump_steps = ctx.landmark_reset_steps or None
    ctx.all_particles = np.stack(ctx.all_particles_list)
    ctx.particle_paths = _reconstruct_particle_paths(
        ctx.position_history, ctx.parent_history
    )
    if allowed_jump_steps:
        ctx.current_path, ctx.current_modes, ctx.current_sources = (
            _select_reachable_cluster_path(
                ctx.position_history,
                ctx.weight_history,
                ctx.parent_history,
                ctx.map_gray,
                ctx.gx_mean,
                ctx.gz_mean,
                ctx.origin_px,
                ctx.scale,
                allowed_jump_steps,
            )
        )
        ctx.sequence_path = ctx.current_path.copy()
        ctx.sequence_modes = ["reset_current"] * len(ctx.current_path)
        ctx.sequence_sources = list(ctx.current_sources)
        ctx.current_reversals = 0
        ctx.sequence_reversals = 0
        ctx.selected_path = ctx.current_path
        ctx.trajectory_modes = ctx.current_modes
        ctx.trajectory_sources = ctx.current_sources
        ctx.selected_mode = "current"
    elif ctx.path_selection == "sequence" or ctx.recorder.paths_enabled:
        ctx.sensor_headings = np.asarray(
            [heading.selected_heading for heading in ctx.step_headings],
            dtype=float,
        )
        ctx.turning_evidence = np.asarray(
            [
                heading.trajectory_movement_type == "turning"
                or heading.movement_type == "turning"
                or (
                    ctx.prepared_motion_posteriors is not None
                    and index < len(ctx.prepared_motion_posteriors)
                    and (
                        ctx.prepared_motion_posteriors[index].turning_probability
                        >= 0.25
                    )
                )
                for index, heading in enumerate(ctx.step_headings)
            ],
            dtype=bool,
        )
        ctx.sequence_path, ctx.sequence_modes, ctx.sequence_sources = (
            _select_sequence_map_path(
                ctx.particle_paths,
                ctx.path_log_score_history[-1],
                ctx.map_gray,
                ctx.gx_mean,
                ctx.gz_mean,
                ctx.origin_px,
                ctx.scale,
                ctx.sensor_headings,
                ctx.turning_evidence,
                allowed_jump_steps,
            )
        )
        ctx.current_path, ctx.current_modes, ctx.current_sources = (
            _select_reachable_cluster_path(
                ctx.position_history,
                ctx.weight_history,
                ctx.parent_history,
                ctx.map_gray,
                ctx.gx_mean,
                ctx.gz_mean,
                ctx.origin_px,
                ctx.scale,
                allowed_jump_steps,
            )
        )
        ctx.sequence_reversals = _unsupported_reversal_count(
            ctx.sequence_path, ctx.sensor_headings, ctx.turning_evidence
        )
        ctx.current_reversals = _unsupported_reversal_count(
            ctx.current_path, ctx.sensor_headings, ctx.turning_evidence
        )
        if ctx.path_selection == "sequence" and (
            ctx.landmark_mode == "ranging"
            or ctx.sequence_reversals < ctx.current_reversals
        ):
            ctx.selected_path = ctx.sequence_path
            ctx.trajectory_modes = ctx.sequence_modes
            ctx.trajectory_sources = ctx.sequence_sources
            ctx.selected_mode = "sequence"
        else:
            ctx.selected_path = ctx.current_path
            ctx.trajectory_modes = ctx.current_modes
            ctx.trajectory_sources = ctx.current_sources
            ctx.selected_mode = "current"
    else:
        ctx.selected_path, ctx.trajectory_modes, ctx.trajectory_sources = (
            _select_reachable_cluster_path(
                ctx.position_history,
                ctx.weight_history,
                ctx.parent_history,
                ctx.map_gray,
                ctx.gx_mean,
                ctx.gz_mean,
                ctx.origin_px,
                ctx.scale,
                allowed_jump_steps,
            )
        )
        ctx.selected_mode = "current"
    _retrofit_landmark_anchor_jumps(ctx)
    if allowed_jump_steps:
        for step in allowed_jump_steps - ctx.landmark_anchor_steps:
            jump_distance = float(
                np.linalg.norm(ctx.selected_path[step] - ctx.selected_path[step - 1])
            )
            if jump_distance > ctx.landmark_max_jump_m + 1e-9:
                raise RuntimeError(
                    "ランドマークreset後の代表軌跡がジャンプ上限を超えました: "
                    f"step={step} distance={jump_distance:.3f}m"
                )
    if ctx.recorder.paths_enabled:
        ctx.recorder.paths.append(
            ParticlePathComparison(
                selected_mode=ctx.selected_mode,
                selected_path=ctx.selected_path.copy(),
                current_path=ctx.current_path.copy(),
                sequence_path=ctx.sequence_path.copy(),
                particle_paths=ctx.particle_paths.copy(),
                current_reversals=ctx.current_reversals,
                sequence_reversals=ctx.sequence_reversals,
            )
        )
    if ctx.recorder.diagnostics_enabled:
        for diagnostic_offset, (mode, source) in enumerate(
            zip(ctx.trajectory_modes[1:], ctx.trajectory_sources[1:], strict=True)
        ):
            ctx.collector_index = ctx.diagnostics_start_index + diagnostic_offset
            ctx.recorder.diagnostics[ctx.collector_index] = replace(
                ctx.recorder.diagnostics[ctx.collector_index],
                trajectory_mode=mode,
                trajectory_source_index=source,
            )
    return (
        ctx.selected_path.tolist(),
        ctx.step_lengths,
        ctx.t_at_steps,
        ctx.all_particles,
        ctx.step_headings,
    )
