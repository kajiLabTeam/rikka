"""全歩を走らせたあとの代表軌跡の確定。どの粒子の足跡を採るかを決める。

役割:
    粒子の位置履歴と親系譜から1本の代表軌跡を選び、呼び出し側へ返す形に整える。
依存元:
    ``landmark/lib`` の相似変換、``particle/lib`` の経路復元・経路選択・
    収束クラスタ化、``ParticleRuntime``。
利用先:
    ``particle.lib.runner`` が歩ループを終えたあとに1回だけ呼び出す。
処理フロー:
    代表軌跡の選び方は、実行条件によって次の順に分岐する。

    1. ``init_mode="uniform"``: 最終時点の最大クラスタの中で経路スコアが最大の
       粒子を選び、その祖先経路を採る（遅延決定）。実在した粒子の足跡なので
       壁を跨がない。全粒子の重み付き平均は複数クラスタの中点を通り壁を
       突き抜けるため使わない。
    2. 確定位置アンカーがある場合: アンカーで区切った区間ごとに選んだ経路を
       連結する。
    3. ランドマーク reset が発生した場合: 祖先経路が不連続になるため、
       時点別の到達可能クラスタ経路（current）を使う。
    4. それ以外: ``path_selection`` が ``sequence`` なら、持続反転が減る場合だけ
       単一祖先経路を採り、減らなければ current へ戻る。

    選択後、``landmark_retrofit`` が有効なら確定アンカー直前の区間だけを相似変換し、
    位置ジャンプを除く。最後に診断へ採用モードを書き戻す。
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
from .convergence import cluster_distribution
from .hard_anchor import hard_segment_reversals, select_hard_segment
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
    """代表軌跡を選び、(軌跡, 歩幅, 時刻, 全粒子列, 歩方位) を返す。

    戻り値はこの順の5要素で、``run_particle_steps()`` の互換形式である。
    """
    allowed_jump_steps = ctx.landmark_reset_steps or None
    ctx.all_particles = np.stack(ctx.all_particles_list)
    ctx.particle_paths = _reconstruct_particle_paths(
        ctx.position_history, ctx.parent_history
    )
    if ctx.init_mode == "uniform":
        _, indices = cluster_distribution(
            ctx.position_history[-1],
            ctx.weight_history[-1],
            ctx.convergence_cell_m,
            ctx.convergence_top_k,
        )
        if not len(indices):
            raise RuntimeError("代表経路を選択できる生存粒子がありません")
        best = int(indices[np.argmax(ctx.path_log_score_history[-1][indices])])
        ctx.selected_path = ctx.particle_paths[best].copy()
        ctx.selected_mode = "uniform_delayed"
        ctx.trajectory_modes = ["uniform_delayed"] * len(ctx.selected_path)
        if ctx.convergence is None:
            print("PF収束: 未収束")
        else:
            convergence = ctx.convergence
            print(
                f"PF収束: K={convergence.step_number}, "
                f"位置=({convergence.dominant_centroid_x_m:.3f}, "
                f"{convergence.dominant_centroid_y_m:.3f}) m, "
                f"最大クラスタ重み比={convergence.dominant_weight_ratio:.4f}"
            )
        ctx.trajectory_sources = [None] * len(ctx.selected_path)
        ctx.current_path = ctx.selected_path.copy()
        ctx.sequence_path = ctx.selected_path.copy()
        ctx.current_reversals = 0
        ctx.sequence_reversals = 0
    elif ctx.hard_anchor_events:
        current, sequence = select_hard_segment(ctx)
        ctx.current_path = np.asarray([*ctx.hard_current_prefix, *current.tolist()[1:]])
        ctx.sequence_path = np.asarray(
            [*ctx.hard_sequence_prefix, *sequence.tolist()[1:]]
        )
        ctx.current_modes = ["hard_segment_current"] * len(ctx.current_path)
        ctx.sequence_modes = ["hard_segment_sequence"] * len(ctx.sequence_path)
        ctx.current_sources = [None] * len(ctx.current_path)
        ctx.sequence_sources = [None] * len(ctx.sequence_path)
        ctx.current_reversals = ctx.hard_current_reversals + hard_segment_reversals(
            ctx, current
        )
        ctx.sequence_reversals = ctx.hard_sequence_reversals + hard_segment_reversals(
            ctx, sequence
        )
        if ctx.path_selection == "sequence":
            ctx.selected_path = ctx.sequence_path
            ctx.trajectory_modes = ctx.sequence_modes
            ctx.trajectory_sources = ctx.sequence_sources
            ctx.selected_mode = "sequence"
        else:
            ctx.selected_path = ctx.current_path
            ctx.trajectory_modes = ctx.current_modes
            ctx.trajectory_sources = ctx.current_sources
            ctx.selected_mode = "current"
    elif allowed_jump_steps:
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
            (ctx.proposal_method == "legacy" and ctx.landmark_mode == "ranging")
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
    if not ctx.hard_anchor_events:
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
