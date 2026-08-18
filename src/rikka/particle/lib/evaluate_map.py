"""particle filter の地図制約解決。

役割:
    提案後の地図制約違反をrecoveryで解決し、必要なら再標本化する。
依存元:
    particle の復旧適用関数、再標本化、角度正規化、ParticleRuntime を利用する。
利用先:
    particle/lib/runner が提案・重み計算の後に呼び出す。
処理フロー:
    reset方式のランドマーク歩では位置を再配置する。それ以外の地図違反時は局所復旧、
    checkpoint再生、fallback、位置保持を順に試し、通常経路ではESSに基づいて
    再標本化して次状態を確定する。
"""

import sys

import numpy as np

from .landmark import meter_walkable_mask, reset_particles_to_landmark
from .proposal import _normalize_angle
from .recovery.apply import (
    _apply_checkpoint_replay,
    _apply_fallback_recovery,
    _apply_local_recovery,
    _hold_on_recovery_failure,
    _try_checkpoint_replay,
    _try_local_recovery,
)
from .resampling import _systematic_resample
from .state import ParticleRuntime


def _reset_recovery_diagnostics(ctx: ParticleRuntime) -> None:
    """現在歩の復旧・再標本化診断を初期状態へ戻す。"""
    ctx.recovery_step_scale = None
    ctx.recovery_cost = None
    ctx.recovery_checkpoint_step = None
    ctx.recovery_replay_steps = 0
    ctx.recovery_candidate_branch_count = 0
    ctx.recovery_selected_branch_count = 0
    ctx.recovery_candidate_headings = None
    ctx.recovery_candidate_valid = None
    ctx.recovery_selected_index = None
    ctx.resampled = False
    ctx.next_path_log_scores = np.empty(0, dtype=float)


def _resample_or_keep(ctx: ParticleRuntime) -> None:
    """正規化済み事後重みを再標本化するか、そのまま採用する。"""
    if ctx.valid_weight_mass <= 0.0:
        raise RuntimeError("内部エラー: recoveryせず粒子重みが全滅しました")
    ctx.posterior_weights /= ctx.valid_weight_mass
    if ctx.ess_after_observation < ctx.resample_ess_ratio * ctx.n_particles:
        ctx.indices = _systematic_resample(ctx.posterior_weights, ctx.rng)
        ctx.particles = ctx.proposed_particles[ctx.indices]
        ctx.heading_correction = ctx.proposed_correction[ctx.indices]
        ctx.heading_drift = ctx.proposed_drift[ctx.indices]
        ctx.stride_scale = ctx.proposed_stride_scale[ctx.indices]
        ctx.motion_state = ctx.proposed_motion_state[ctx.indices]
        if ctx.effective_heading_rejuvenation_sigma > 0.0:
            ctx.heading_drift = _normalize_angle(
                ctx.heading_drift
                + ctx.rng.normal(
                    0,
                    ctx.effective_heading_rejuvenation_sigma,
                    ctx.n_particles,
                )
            )
        ctx.effective_rejuvenation_sigma = (
            max(ctx.stride_scale_rejuvenation_sigma, 0.02)
            if ctx.adaptive_stride_state
            else ctx.stride_scale_rejuvenation_sigma
        )
        if ctx.effective_rejuvenation_sigma > 0.0:
            ctx.stride_scale = np.clip(
                ctx.stride_scale
                + ctx.stride_rng.normal(
                    0.0,
                    ctx.effective_rejuvenation_sigma,
                    ctx.n_particles,
                ),
                ctx.effective_stride_scale_min,
                ctx.effective_stride_scale_max,
            )
        ctx.weights = np.full(ctx.n_particles, 1.0 / ctx.n_particles)
        ctx.parent_indices = ctx.indices
        ctx.effective_step_lengths_for_diagnostics = ctx.sl[ctx.indices]
        ctx.next_path_log_scores = ctx.candidate_path_log_scores[ctx.indices]
        ctx.resampled = True
        return

    ctx.particles = ctx.proposed_particles
    ctx.heading_correction = ctx.proposed_correction
    ctx.heading_drift = ctx.proposed_drift
    ctx.stride_scale = ctx.proposed_stride_scale
    ctx.motion_state = ctx.proposed_motion_state
    ctx.weights = ctx.posterior_weights
    ctx.parent_indices = np.arange(ctx.n_particles, dtype=int)
    ctx.next_path_log_scores = ctx.candidate_path_log_scores


def _reset_to_landmark(ctx: ParticleRuntime) -> None:
    """現在歩のランドマーク周辺へ位置だけを再配置し、履歴重みを初期化する。"""
    if ctx.landmark_xy is None:
        raise RuntimeError("内部エラー: reset対象ランドマークがありません。")

    if ctx.landmark_before_position is None:
        raise RuntimeError("内部エラー: reset前の代表位置がありません。")
    reset_origin = np.asarray(ctx.landmark_before_position, dtype=float)

    def is_walkable(points: np.ndarray) -> np.ndarray:
        walkable = meter_walkable_mask(
            points,
            ctx.map_gray,
            ctx.gx_mean,
            ctx.gz_mean,
            ctx.origin_px,
            ctx.scale,
        )
        within_jump = (
            np.linalg.norm(points - reset_origin, axis=1) <= ctx.landmark_max_jump_m
        )
        return np.asarray(walkable & within_jump, dtype=bool)

    ctx.particles = reset_particles_to_landmark(
        ctx.n_particles,
        ctx.landmark_xy,
        ctx.landmark_reset_sigma_m,
        ctx.rng,
        is_walkable,
    )
    ctx.heading_correction = ctx.proposed_correction
    ctx.heading_drift = _normalize_angle(
        ctx.proposed_drift
        + ctx.rng.normal(
            0.0,
            ctx.landmark_reset_heading_sigma,
            ctx.n_particles,
        )
    )
    ctx.stride_scale = ctx.proposed_stride_scale
    ctx.motion_state = ctx.proposed_motion_state
    ctx.weights = np.full(ctx.n_particles, 1.0 / ctx.n_particles)
    ctx.posterior_weights = ctx.weights.copy()
    ctx.parent_indices = np.arange(ctx.n_particles, dtype=int)
    ctx.next_path_log_scores = np.zeros(ctx.n_particles, dtype=float)
    ctx.valid_transition = is_walkable(ctx.particles)
    ctx.valid_count = int(np.count_nonzero(ctx.valid_transition))
    ctx.valid_weight_mask = ctx.valid_transition.copy()
    ctx.valid_weight_count = ctx.valid_count
    ctx.valid_weight_mass = 1.0
    ctx.ess_after_observation = float(ctx.n_particles)
    ctx.effective_step_lengths_for_diagnostics = ctx.sl.copy()
    ctx.resampled = True
    ctx.recovery_attempted = False
    ctx.recovery_mode = "landmark_reset"
    ctx.recovery_valid_count = ctx.valid_count
    ctx.landmark_applied = True
    ctx.landmark_reset_steps.add(ctx.step_number)


def _landmark_reset_requested(ctx: ParticleRuntime) -> bool:
    """現在のmodeと粒子群の広がりからresetが必要かを返す。

    reset と hybrid のどちらも「reset のばら撒き幅より誤差が十分大きい」ことを求める。
    誤差が ``landmark_reset_sigma_m`` と同程度のときに撒き直すと不確かさが増え、
    代表軌跡が往復して折り返す。hybrid はさらに「観測尤度では届かないほど遠い」
    ことも求め、粒子群の広がりで届く範囲は observation に任せる。
    """
    if ctx.landmark_mode not in {"reset", "hybrid"}:
        return False
    if ctx.landmark_before_position is None or ctx.landmark_xy is None:
        return False
    distance = float(
        np.linalg.norm(
            np.asarray(ctx.landmark_before_position) - np.asarray(ctx.landmark_xy)
        )
    )
    if distance <= ctx.landmark_reset_min_distance_m:
        return False
    if ctx.landmark_mode == "reset":
        return True
    if ctx.landmark_position_spread_rms_m is None:
        return False
    spread = max(ctx.landmark_position_spread_rms_m, 1e-9)
    return distance > ctx.landmark_reset_spread_ratio * spread


def _landmark_reset_within_limit(ctx: ParticleRuntime) -> bool:
    """reset先までの代表距離が安全上限以内かを返す。"""
    if ctx.landmark_before_position is None or ctx.landmark_xy is None:
        return False
    distance = float(
        np.linalg.norm(
            np.asarray(ctx.landmark_before_position) - np.asarray(ctx.landmark_xy)
        )
    )
    if distance <= ctx.landmark_max_jump_m:
        return True
    beacon_id = (
        "unknown"
        if ctx.landmark_detection is None
        else ctx.landmark_detection.beacon_id
    )
    print(
        f"警告: ランドマーク {beacon_id} へのreset距離 {distance:.2f}mが "
        f"上限 {ctx.landmark_max_jump_m:.2f}mを超えたためスキップします。",
        file=sys.stderr,
    )
    return False


def resolve_map_constraints(ctx: ParticleRuntime) -> None:
    """地図制約違反を復旧し、通常粒子はESSに応じて再標本化する。"""
    _reset_recovery_diagnostics(ctx)
    if ctx.landmark_detection is not None and _landmark_reset_requested(ctx):
        if _landmark_reset_within_limit(ctx):
            _reset_to_landmark(ctx)
            return
    if not ctx.recovery_attempted:
        _resample_or_keep(ctx)
        return

    if _try_local_recovery(ctx):
        _apply_local_recovery(ctx)
        return
    if _try_checkpoint_replay(ctx):
        _apply_checkpoint_replay(ctx)
        return
    if _apply_fallback_recovery(ctx):
        return
    _hold_on_recovery_failure(ctx)
