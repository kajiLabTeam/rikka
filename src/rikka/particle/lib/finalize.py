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


def finalize(
    ctx: ParticleRuntime,
) -> tuple[
    list[list[float]],
    list[float],
    list[float],
    np.ndarray,
    list[StepHeading],
]:
    ctx.all_particles = np.stack(ctx.all_particles_list)
    ctx.particle_paths = _reconstruct_particle_paths(
        ctx.position_history, ctx.parent_history
    )
    if ctx.path_selection == "sequence" or ctx.recorder.paths_enabled:
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
            )
        )
        ctx.sequence_reversals = _unsupported_reversal_count(
            ctx.sequence_path, ctx.sensor_headings, ctx.turning_evidence
        )
        ctx.current_reversals = _unsupported_reversal_count(
            ctx.current_path, ctx.sensor_headings, ctx.turning_evidence
        )
        if (
            ctx.path_selection == "sequence"
            and ctx.sequence_reversals < ctx.current_reversals
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
            )
        )
        ctx.selected_mode = "current"
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
