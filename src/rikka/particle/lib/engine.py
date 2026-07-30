"""パーティクルフィルタの入力準備と粒子更新ループ。

役割:
    PDRの歩行ステップを粒子群へ反映し、地図制約、再標本化、復旧、診断生成を
    順序を保って統括する。
依存元:
    ``config`` と ``pdr.particle_api`` から設定・歩行推定APIを取得し、同じ
    ``particle`` パッケージの運動、地図、再標本化、復旧、経路モジュールを使う。
利用先:
    ``rikka.analyze.particle_filter`` の互換facadeを通じてPDR pipelineとCLIから
    使用される。
処理フロー:
    入力を検証して3系列の乱数生成器を初期化し、各歩の提案・観測・復旧・履歴更新を
    行い、各時点の到達可能な有力粒子クラスタから代表軌跡と診断結果を返す。
"""

from dataclasses import replace
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from ...common.lib.models import (
    StepHeading,
    StepMotionEvidence,
    StepMotionPosterior,
    StepSegment,
)
from ...common.lib.pdr_math import (
    _validate_forward_heading_source as validate_forward_heading_source,
)
from ...common.lib.pdr_math import (
    _validate_motion_heading_correction as validate_motion_heading_correction,
)
from ...common.lib.pdr_math import (
    _validate_sidestep_heading_source as validate_sidestep_heading_source,
)
from ...common.lib.pdr_math import (
    _validate_sidestep_smoothing as validate_sidestep_smoothing,
)
from ...common.lib.pdr_math import (
    _validate_sidestep_suspect_mode as validate_sidestep_suspect_mode,
)
from ...common.lib.time_utils import _step_output_time as step_output_time
from ...common.lib.validation import (
    validate_non_negative_parameter,
    validate_positive_parameter,
)
from ...config import (
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    INITIAL_DIRECTION,
    PF_HEADING_DRIFT_RETENTION,
    PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    PF_NUM_PARTICLES,
    PF_PATH_SELECTION,
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
from ...particle.lib.map_constraints import (
    _evaluate_particle_transitions,
    _normalize_floormap_gray,
)
from ...particle.lib.proposal import (
    _MOTION_FORWARD,
    _motion_state_headings,
    _motion_state_likelihoods,
    _normalize_angle,
    _sample_motion_states,
)
from ...particle.lib.recorder import (
    ParticleFilterStepDiagnostics,
    ParticlePathComparison,
    ParticleRecorder,
    ParticleStepStages,
)
from ...particle.lib.recovery.local import (
    _generate_recovery_candidates,
    _replay_from_checkpoint,
)
from ...particle.lib.resampling import (
    _effective_sample_size,
    _systematic_resample,
)
from ...particle.lib.state import ParticleHistory, ParticleState
from ...particle.lib.weighting import weight
from ...pdr.lib.heading.device_orientation import estimate_device_orientation_mode
from ...pdr.lib.heading.motion import resolve_motion_heading_correction
from ...pdr.lib.heading.resolver import resolve_step_heading
from ...pdr.lib.motion_state.clustering import smooth_step_headings
from ...pdr.lib.motion_state.evidence import (
    build_particle_motion_headings,
    build_step_motion_evidences,
)
from ...pdr.lib.motion_state.heading_policy import stabilize_trajectory_headings
from ...pdr.lib.motion_state.step_motion import estimate_step_motion
from ...pdr.lib.step_length import (
    estimate_initial_forward_angle,
    estimate_step_length,
    estimate_step_length_forward,
)
from .diagnostics import _build_step_diagnostics
from .paths import (
    _reconstruct_particle_paths,
    _select_reachable_cluster_path,
    _select_sequence_map_path,
    _unsupported_reversal_count,
)


def _adaptive_heading_rejuvenation_sigma(
    base_sigma: float,
    motion_reliability: float,
) -> float:
    """高信頼な移動方位観測がある記録では再標本化ノイズを弱める。"""
    reliability = (
        float(np.clip(motion_reliability, 0.0, 1.0))
        if np.isfinite(motion_reliability)
        else 0.0
    )
    high_reliability_excess = max(0.0, reliability - 0.90)
    high_reliability_scale = 4.0 / 9.0
    scale = float(
        np.clip(
            1.0 - (1.0 - high_reliability_scale) * high_reliability_excess / 0.02,
            high_reliability_scale,
            1.0,
        )
    )
    return base_sigma * scale


def _run_particle_steps(
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
    prepared_particle_motion_headings: tuple[float | None, ...] | None = None,
    prepared_motion_posteriors: tuple[StepMotionPosterior, ...] | None = None,
    seed: int | None = None,
    heading_drift_retention: float = PF_HEADING_DRIFT_RETENTION,
    resample_ess_ratio: float = PF_RESAMPLE_ESS_RATIO,
    rejuvenation_sigma_heading: float = PF_REJUVENATION_SIGMA_HEADING,
    recovery_valid_ratio: float = PF_RECOVERY_VALID_RATIO,
    recovery_heading_sigma: float = PF_RECOVERY_HEADING_SIGMA,
    recovery_max_attempts: int = PF_RECOVERY_MAX_ATTEMPTS,
    diagnostics_collector: list[ParticleFilterStepDiagnostics] | None = None,
    stage_collector: list[ParticleStepStages] | None = None,
    path_comparison_collector: list[ParticlePathComparison] | None = None,
    preserve_recovery_branches: bool = False,
    motion_predictive_weight_power: float = PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    path_selection: str = PF_PATH_SELECTION,
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
        stage_collector: 指定時に可視化用の段階別粒子状態を追記するリスト
        path_comparison_collector: 指定時に代表軌跡候補を1件追記するリスト
        preserve_recovery_branches: recoveryで複数の経路方位族を保護するか
        motion_predictive_weight_power: 運動状態の予測尤度を重みに掛ける指数
        path_selection: 代表軌跡を従来方式または単一祖先系列から選ぶ方式

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
    motion_predictive_weight_power = validate_non_negative_parameter(
        "motion_predictive_weight_power", motion_predictive_weight_power
    )
    if path_selection not in {"current", "sequence"}:
        raise ValueError("path_selection は current または sequence を指定してください")
    if recovery_max_attempts <= 0:
        raise ValueError("recovery_max_attempts は正の整数を指定してください")
    adaptive_stride_state = bool(
        prepared_motion_posteriors is not None
        and any(
            posterior.length_std_m / max(posterior.length_mean_m, 1e-6) >= 0.18
            for posterior in prepared_motion_posteriors
        )
    )
    effective_stride_scale_min = (
        min(stride_scale_min, 0.6) if adaptive_stride_state else stride_scale_min
    )
    effective_stride_scale_max = (
        max(stride_scale_max, 1.5) if adaptive_stride_state else stride_scale_max
    )
    effective_stride_init_sigma = (
        max(stride_scale_init_sigma, 0.12)
        if adaptive_stride_state
        else stride_scale_init_sigma
    )
    rng = np.random.default_rng(seed)
    stride_rng = np.random.default_rng(
        None if seed is None else np.random.SeedSequence([seed, 0x53545249])
    )
    motion_rng = np.random.default_rng(
        None if seed is None else np.random.SeedSequence([seed, 0x4D4F544E])
    )

    # フロアマップをグレースケールで読み込み
    map_gray = _normalize_floormap_gray(mpimg.imread(Path(floormap_path)))
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
        + stride_rng.normal(0.0, effective_stride_init_sigma, n_particles),
        effective_stride_scale_min,
        effective_stride_scale_max,
    )
    weights = np.ones(n_particles) / n_particles
    initial_state = ParticleState(
        particles,
        heading_correction,
        heading_drift,
        motion_state,
        stride_scale,
        weights,
    )

    step_lengths: list[float] = []
    t_at_steps: list[float] = []
    history = ParticleHistory()
    history.append(initial_state, np.zeros(n_particles, dtype=float))
    position_history = history.positions
    heading_correction_history = history.heading_corrections
    heading_drift_history = history.heading_drifts
    motion_state_history = history.motion_states
    stride_scale_history = history.stride_scales
    weight_history = history.weights
    path_log_score_history = history.path_log_scores
    parent_history = history.parents
    all_particles_list: list[np.ndarray] = [particles.copy()]  # ステップ0（原点）
    step_headings: list[StepHeading] = []
    healthy_checkpoint_steps = [0]
    recorder = ParticleRecorder(
        diagnostics_collector,
        stage_collector,
        path_comparison_collector,
    )
    diagnostics_start_index = len(recorder.diagnostics)
    stages_start_index = len(recorder.stages)
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
    if prepared_motion_posteriors is not None and not using_prepared_steps:
        raise ValueError(
            "prepared_motion_posteriors は prepared step 一式と同時に指定してください"
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
        if prepared_motion_posteriors is not None:
            if len(prepared_motion_posteriors) != len(stabilized_step_headings):
                raise ValueError(
                    "prepared_motion_posteriors と prepared_step_headings の"
                    "長さが一致しません"
                )
    particle_motion_headings = (
        build_particle_motion_headings(stabilized_step_headings)
        if prepared_particle_motion_headings is None
        else prepared_particle_motion_headings
    )
    if len(particle_motion_headings) != len(stabilized_step_headings):
        raise ValueError(
            "prepared_particle_motion_headings と prepared_step_headings の"
            "長さが一致しません"
        )
    recording_motion_reliability = (
        float(np.median([evidence.motion_reliability for evidence in motion_evidences]))
        if motion_evidences
        else 0.0
    )
    effective_heading_rejuvenation_sigma = _adaptive_heading_rejuvenation_sigma(
        rejuvenation_sigma_heading,
        recording_motion_reliability,
    )

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
        motion_posterior = (
            prepared_motion_posteriors[step_number - 1]
            if prepared_motion_posteriors is not None
            else None
        )
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
        diagnostic_motion_state_before = motion_state_before
        weights_before = weights.copy()
        path_log_scores_before = path_log_score_history[-1].copy()
        ess_before_observation = _effective_sample_size(weights_before)

        # 通常ドリフトだけを平均回帰させ、recovery補正は独立に保持する。
        proposed_correction = heading_correction_before
        heading_process_sigma = (
            max(sigma_heading, 0.15 * motion_posterior.heading_std)
            if motion_posterior is not None
            else sigma_heading
        )
        proposed_drift = _normalize_angle(
            heading_drift_retention * heading_drift_before
            + rng.normal(0, heading_process_sigma, n_particles)
        )
        observation_likelihoods = _motion_state_likelihoods(motion_evidence)
        proposed_motion_state, state_predictive_likelihoods = _sample_motion_states(
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
        relative_length_uncertainty = (
            motion_posterior.length_std_m / max(motion_posterior.length_mean_m, 1e-6)
            if motion_posterior is not None
            else 0.0
        )
        adaptive_recovery_scale = (
            motion_posterior is not None and relative_length_uncertainty >= 0.18
        )
        step_stride_process_sigma = (
            max(
                stride_scale_process_sigma,
                min(0.05, 0.12 * relative_length_uncertainty),
            )
            if motion_posterior is not None
            else stride_scale_process_sigma
        )
        proposed_stride_scale = np.clip(
            stride_scale_prior_mean
            + stride_scale_retention * (stride_scale_before - stride_scale_prior_mean)
            + stride_rng.normal(0.0, step_stride_process_sigma, n_particles),
            effective_stride_scale_min,
            effective_stride_scale_max,
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
        effective_step_lengths_for_diagnostics = sl.copy()

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
        if motion_posterior is None:
            stride_observation_likelihood = np.ones(n_particles, dtype=float)
        else:
            stride_prior_sigma = float(
                np.clip(1.5 * relative_length_uncertainty, 0.10, 0.30)
            )
            stride_observation_likelihood = np.exp(
                -0.5
                * np.square(
                    (proposed_stride_scale - stride_scale_prior_mean)
                    / stride_prior_sigma
                )
            )
        posterior_weights = weight(
            weights_before,
            valid_transition,
            stride_observation_likelihood,
            state_predictive_likelihoods,
            motion_predictive_weight_power,
        )
        posterior_weights_for_stages = (
            posterior_weights.copy() if recorder.stages_enabled else None
        )
        with np.errstate(divide="ignore"):
            observation_log_likelihood = np.log(stride_observation_likelihood)
            if motion_predictive_weight_power > 0.0:
                observation_log_likelihood += motion_predictive_weight_power * np.log(
                    state_predictive_likelihoods
                )
        candidate_path_log_scores = np.where(
            valid_transition,
            path_log_scores_before + observation_log_likelihood,
            -np.inf,
        )
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
        recovery_candidate_branch_count = 0
        recovery_selected_branch_count = 0
        recovery_candidate_headings: np.ndarray | None = None
        recovery_candidate_valid: np.ndarray | None = None
        recovery_selected_index: np.ndarray | None = None
        resampled = False
        next_path_log_scores: np.ndarray

        if recovery_attempted:
            allow_turn_candidates = bool(
                (
                    motion_posterior is not None
                    and motion_posterior.turning_probability >= 0.25
                )
                or step_heading.trajectory_movement_type == "turning"
                or step_heading.movement_type == "turning"
                or "sidestep"
                in (step_heading.trajectory_movement_type or step_heading.movement_type)
                or (
                    bool(step_headings)
                    and (
                        step_headings[-1].trajectory_movement_type == "turning"
                        or step_headings[-1].movement_type == "turning"
                    )
                )
            )
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
                allow_turn_candidates=allow_turn_candidates,
                preserve_route_branches=preserve_recovery_branches,
                allow_stride_adaptation=adaptive_recovery_scale,
                stride_scale_min=effective_stride_scale_min,
                stride_scale_max=effective_stride_scale_max,
                capture_candidates=recorder.stages_enabled,
            )
            recovery_attempts = (
                recovery.attempts
                if recovery is not None
                else min(recovery_max_attempts, 2 if allow_turn_candidates else 1)
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
                        allow_stride_adaptation=adaptive_recovery_scale,
                        stride_scale_min=effective_stride_scale_min,
                        stride_scale_max=effective_stride_scale_max,
                        capture_candidates=recorder.stages_enabled,
                    )
                    recovery_attempts += 1
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
                        preserve_route_branches=preserve_recovery_branches,
                        allow_stride_adaptation=adaptive_recovery_scale,
                        stride_scale_min=effective_stride_scale_min,
                        stride_scale_max=effective_stride_scale_max,
                        capture_candidates=recorder.stages_enabled,
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
                        effective_step_lengths_for_diagnostics = np.zeros(
                            n_particles, dtype=float
                        )
                        next_path_log_scores = path_log_scores_before.copy()
                        recovery_mode = "failed_hold"
                        recovery_attempts += min(recovery_max_attempts, 2)
                    else:
                        particles = fallback_recovery.particles
                        heading_correction = fallback_recovery.heading_correction
                        heading_drift = fallback_recovery.heading_drift
                        stride_scale = fallback_recovery.stride_scale
                        motion_state = fallback_recovery.motion_state
                        weights = np.full(n_particles, 1.0 / n_particles)
                        parent_indices = fallback_recovery.parent_indices
                        effective_step_lengths_for_diagnostics = np.linalg.norm(
                            particles - particles_before[parent_indices], axis=1
                        )
                        next_path_log_scores = (
                            path_log_scores_before[parent_indices]
                            + fallback_recovery.path_log_score_delta
                        )
                        recovery_mode = f"fallback_{fallback_recovery.mode}"
                        recovery_valid_count = fallback_recovery.valid_count
                        recovery_attempts += fallback_recovery.attempts
                        recovery_heading_delta_deg = fallback_recovery.heading_delta_deg
                        recovery_step_scale = fallback_recovery.step_scale
                        recovery_cost = fallback_recovery.mean_cost
                        recovery_candidate_branch_count = int(
                            np.unique(fallback_recovery.route_branch_ids).size
                        )
                        recovery_selected_branch_count = recovery_candidate_branch_count
                        recovery_candidate_headings = (
                            fallback_recovery.candidate_headings
                        )
                        recovery_candidate_valid = fallback_recovery.candidate_valid
                        recovery_selected_index = (
                            fallback_recovery.selected_candidate_indices
                        )
                        resampled = True
                else:
                    recovery = replay_result.recovery
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
                    path_log_score_history = path_log_score_history[
                        : checkpoint_step + 1
                    ]
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
                        path_log_score_history.append(
                            path_log_score_history[checkpoint_step][
                                recovery.parent_indices
                            ].copy()
                        )
                    if recorder.diagnostics_enabled:
                        replay_weights = np.full(n_particles, 1.0 / n_particles)
                        for replay_offset in range(replay_depth - 1):
                            history_step = checkpoint_step + replay_offset + 1
                            replay_parent_indices = (
                                recovery.parent_indices
                                if replay_offset == 0
                                else np.arange(n_particles, dtype=int)
                            )
                            replay_motion_state_before = (
                                motion_state_history[checkpoint_step]
                                if replay_offset == 0
                                else recovery.motion_state
                            )
                            replay_previous_positions = (
                                position_history[checkpoint_step][
                                    recovery.parent_indices
                                ]
                                if replay_offset == 0
                                else replay_result.replay_positions[replay_offset - 1]
                            )
                            replay_effective_lengths = np.linalg.norm(
                                replay_result.replay_positions[replay_offset]
                                - replay_previous_positions,
                                axis=1,
                            )
                            collector_index = diagnostics_start_index + history_step - 1
                            recorder.diagnostics[collector_index] = (
                                _build_step_diagnostics(
                                    step_number=history_step,
                                    step_time=t_at_steps[history_step - 1],
                                    valid_count=n_particles,
                                    n_particles=n_particles,
                                    valid_weight_count=n_particles,
                                    valid_weight_mass=1.0,
                                    ess_before_observation=float(n_particles),
                                    ess_after_observation=float(n_particles),
                                    ess_after_resampling=float(n_particles),
                                    weights=replay_weights,
                                    particles=replay_result.replay_positions[
                                        replay_offset
                                    ],
                                    heading_drift=recovery.heading_drift,
                                    heading_correction=recovery.heading_correction,
                                    stride_scale=recovery.stride_scale,
                                    effective_step_lengths=replay_effective_lengths,
                                    parent_indices=replay_parent_indices,
                                    resampled=replay_offset == 0,
                                    motion_state=recovery.motion_state,
                                    motion_state_before=replay_motion_state_before,
                                    motion_evidence=motion_evidences[history_step - 1],
                                    recovery_attempted=True,
                                    recovery_mode="checkpoint_replayed",
                                    recovery_valid_count=recovery.valid_count,
                                    recovery_attempts=recovery_attempts,
                                    recovery_heading_delta_deg=(
                                        recovery.heading_delta_deg
                                    ),
                                    recovery_step_scale=recovery.step_scale,
                                    recovery_cost=recovery.mean_cost,
                                    recovery_checkpoint_step=checkpoint_step,
                                    recovery_replay_steps=replay_depth,
                                    recovery_candidate_branch_count=int(
                                        np.unique(recovery.route_branch_ids).size
                                    ),
                                    recovery_selected_branch_count=int(
                                        np.unique(recovery.route_branch_ids).size
                                    ),
                                )
                            )
                    if recorder.stages_enabled:
                        replay_weights = np.full(n_particles, 1.0 / n_particles)
                        replay_offsets = _normalize_angle(
                            recovery.heading_correction + recovery.heading_drift
                        )
                        for replay_offset in range(replay_depth - 1):
                            history_step = checkpoint_step + replay_offset + 1
                            replay_heading = step_headings[history_step - 1]
                            replay_parent_indices = (
                                recovery.parent_indices
                                if replay_offset == 0
                                else np.arange(n_particles, dtype=int)
                            )
                            replay_before_positions = (
                                position_history[checkpoint_step][
                                    recovery.parent_indices
                                ]
                                if replay_offset == 0
                                else replay_result.replay_positions[replay_offset - 1]
                            )
                            replay_after_positions = replay_result.replay_positions[
                                replay_offset
                            ]
                            replay_step_lengths = np.linalg.norm(
                                replay_after_positions - replay_before_positions,
                                axis=1,
                            )
                            replay_sensor_heading = replay_heading.selected_heading
                            replay_proposed_headings = (
                                np.full(
                                    n_particles,
                                    float(replay_sensor_heading or 0.0),
                                )
                                + replay_offsets
                            )
                            collector_index = stages_start_index + history_step - 1
                            recorder.stages[collector_index] = ParticleStepStages(
                                step=history_step,
                                timestamp_s=t_at_steps[history_step - 1],
                                sensor_heading=replay_sensor_heading,
                                sensor_yaw_delta=replay_heading.yaw_delta,
                                movement_type=(
                                    replay_heading.trajectory_movement_type
                                    or replay_heading.movement_type
                                ),
                                deterministic_step_length_m=step_lengths[
                                    history_step - 1
                                ],
                                before_positions=replay_before_positions.copy(),
                                before_offsets=replay_offsets.copy(),
                                before_weights=replay_weights.copy(),
                                before_motion_state=recovery.motion_state.copy(),
                                proposed_positions=replay_after_positions.copy(),
                                proposed_headings=replay_proposed_headings.copy(),
                                proposed_step_lengths=replay_step_lengths.copy(),
                                proposed_motion_state=recovery.motion_state.copy(),
                                valid_transition=np.ones(n_particles, dtype=bool),
                                posterior_weights=replay_weights.copy(),
                                ess_before_observation=float(n_particles),
                                ess_after_observation=float(n_particles),
                                parent_indices=replay_parent_indices.copy(),
                                resampled=replay_offset == 0,
                                recovery_mode="checkpoint_replayed",
                                recovery_candidate_headings=(
                                    recovery.candidate_headings.copy()
                                    if replay_offset == 0
                                    and recovery.candidate_headings is not None
                                    else None
                                ),
                                recovery_candidate_valid=(
                                    recovery.candidate_valid.copy()
                                    if replay_offset == 0
                                    and recovery.candidate_valid is not None
                                    else None
                                ),
                                recovery_selected_index=(
                                    recovery.selected_candidate_indices.copy()
                                    if replay_offset == 0
                                    and recovery.selected_candidate_indices is not None
                                    else None
                                ),
                                after_positions=replay_after_positions.copy(),
                                after_offsets=replay_offsets.copy(),
                                after_weights=replay_weights.copy(),
                                after_motion_state=recovery.motion_state.copy(),
                            )
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
                    diagnostic_motion_state_before = (
                        recovery.motion_state
                        if replay_depth > 1
                        else motion_state_history[checkpoint_step]
                    )
                    effective_step_lengths_for_diagnostics = np.linalg.norm(
                        particles
                        - (
                            replay_result.replay_positions[-2]
                            if replay_depth > 1
                            else position_history[checkpoint_step][
                                recovery.parent_indices
                            ]
                        ),
                        axis=1,
                    )
                    next_path_log_scores = (
                        path_log_score_history[checkpoint_step][
                            recovery.parent_indices
                        ].copy()
                        + recovery.path_log_score_delta
                    )
                    recovery_mode = recovery.mode
                    recovery_valid_count = recovery.valid_count
                    recovery_heading_delta_deg = recovery.heading_delta_deg
                    recovery_step_scale = recovery.step_scale
                    recovery_cost = recovery.mean_cost
                    recovery_checkpoint_step = checkpoint_step
                    recovery_replay_steps = replay_depth
                    recovery_candidate_branch_count = int(
                        np.unique(recovery.route_branch_ids).size
                    )
                    recovery_selected_branch_count = recovery_candidate_branch_count
                    recovery_candidate_headings = recovery.candidate_headings
                    recovery_candidate_valid = recovery.candidate_valid
                    recovery_selected_index = recovery.selected_candidate_indices
                    resampled = True
            else:
                particles = recovery.particles
                heading_correction = recovery.heading_correction
                heading_drift = recovery.heading_drift
                stride_scale = recovery.stride_scale
                motion_state = recovery.motion_state
                weights = np.full(n_particles, 1.0 / n_particles)
                parent_indices = recovery.parent_indices
                effective_step_lengths_for_diagnostics = np.linalg.norm(
                    particles - particles_before[parent_indices], axis=1
                )
                next_path_log_scores = (
                    path_log_scores_before[parent_indices]
                    + recovery.path_log_score_delta
                )
                recovery_mode = recovery.mode
                recovery_valid_count = recovery.valid_count
                recovery_heading_delta_deg = recovery.heading_delta_deg
                recovery_step_scale = recovery.step_scale
                recovery_cost = recovery.mean_cost
                recovery_candidate_branch_count = int(
                    np.unique(recovery.route_branch_ids).size
                )
                recovery_selected_branch_count = recovery_candidate_branch_count
                recovery_candidate_headings = recovery.candidate_headings
                recovery_candidate_valid = recovery.candidate_valid
                recovery_selected_index = recovery.selected_candidate_indices
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
                if effective_heading_rejuvenation_sigma > 0.0:
                    heading_drift = _normalize_angle(
                        heading_drift
                        + rng.normal(
                            0,
                            effective_heading_rejuvenation_sigma,
                            n_particles,
                        )
                    )
                effective_rejuvenation_sigma = (
                    max(stride_scale_rejuvenation_sigma, 0.02)
                    if adaptive_stride_state
                    else stride_scale_rejuvenation_sigma
                )
                if effective_rejuvenation_sigma > 0.0:
                    stride_scale = np.clip(
                        stride_scale
                        + stride_rng.normal(
                            0.0,
                            effective_rejuvenation_sigma,
                            n_particles,
                        ),
                        effective_stride_scale_min,
                        effective_stride_scale_max,
                    )
                weights = np.full(n_particles, 1.0 / n_particles)
                parent_indices = indices
                effective_step_lengths_for_diagnostics = sl[indices]
                next_path_log_scores = candidate_path_log_scores[indices]
                resampled = True
            else:
                particles = proposed_particles
                heading_correction = proposed_correction
                heading_drift = proposed_drift
                stride_scale = proposed_stride_scale
                motion_state = proposed_motion_state
                weights = posterior_weights
                parent_indices = np.arange(n_particles, dtype=int)
                next_path_log_scores = candidate_path_log_scores

        ess_after_resampling = _effective_sample_size(weights)

        position_history.append(particles.copy())
        heading_correction_history.append(heading_correction.copy())
        heading_drift_history.append(heading_drift.copy())
        motion_state_history.append(motion_state.copy())
        stride_scale_history.append(stride_scale.copy())
        weight_history.append(weights.copy())
        path_log_score_history.append(next_path_log_scores.copy())
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
        if recorder.diagnostics_enabled:
            recorder.diagnostics.append(
                _build_step_diagnostics(
                    step_number=step_number,
                    step_time=step_time,
                    valid_count=valid_count,
                    n_particles=n_particles,
                    valid_weight_count=valid_weight_count,
                    valid_weight_mass=valid_weight_mass,
                    ess_before_observation=ess_before_observation,
                    ess_after_observation=ess_after_observation,
                    ess_after_resampling=ess_after_resampling,
                    weights=weights,
                    particles=particles,
                    heading_drift=heading_drift,
                    heading_correction=heading_correction,
                    stride_scale=stride_scale,
                    effective_step_lengths=effective_step_lengths_for_diagnostics,
                    parent_indices=parent_indices,
                    resampled=resampled,
                    motion_state=motion_state,
                    motion_state_before=diagnostic_motion_state_before,
                    motion_evidence=motion_evidence,
                    recovery_attempted=recovery_attempted,
                    recovery_mode=recovery_mode,
                    recovery_valid_count=recovery_valid_count,
                    recovery_attempts=recovery_attempts,
                    recovery_heading_delta_deg=recovery_heading_delta_deg,
                    recovery_step_scale=recovery_step_scale,
                    recovery_cost=recovery_cost,
                    recovery_checkpoint_step=recovery_checkpoint_step,
                    recovery_replay_steps=recovery_replay_steps,
                    recovery_candidate_branch_count=recovery_candidate_branch_count,
                    recovery_selected_branch_count=recovery_selected_branch_count,
                )
            )
        if recorder.stages_enabled:
            assert posterior_weights_for_stages is not None
            recorder.stages.append(
                ParticleStepStages(
                    step=step_number,
                    timestamp_s=step_time,
                    sensor_heading=step_heading.selected_heading,
                    sensor_yaw_delta=step_heading.yaw_delta,
                    movement_type=(
                        step_heading.trajectory_movement_type
                        or step_heading.movement_type
                    ),
                    deterministic_step_length_m=sl_det,
                    before_positions=particles_before.copy(),
                    before_offsets=_normalize_angle(
                        heading_correction_before + heading_drift_before
                    ).copy(),
                    before_weights=weights_before.copy(),
                    before_motion_state=motion_state_before.copy(),
                    proposed_positions=proposed_particles.copy(),
                    proposed_headings=_normalize_angle(theta).copy(),
                    proposed_step_lengths=sl.copy(),
                    proposed_motion_state=proposed_motion_state.copy(),
                    valid_transition=valid_transition.copy(),
                    posterior_weights=posterior_weights_for_stages.copy(),
                    ess_before_observation=ess_before_observation,
                    ess_after_observation=ess_after_observation,
                    parent_indices=parent_indices.copy(),
                    resampled=resampled,
                    recovery_mode=recovery_mode,
                    recovery_candidate_headings=(
                        recovery_candidate_headings.copy()
                        if recovery_candidate_headings is not None
                        else None
                    ),
                    recovery_candidate_valid=(
                        recovery_candidate_valid.copy()
                        if recovery_candidate_valid is not None
                        else None
                    ),
                    recovery_selected_index=(
                        recovery_selected_index.copy()
                        if recovery_selected_index is not None
                        else None
                    ),
                    after_positions=particles.copy(),
                    after_offsets=_normalize_angle(
                        heading_correction + heading_drift
                    ).copy(),
                    after_weights=weights.copy(),
                    after_motion_state=motion_state.copy(),
                )
            )

    all_particles = np.stack(all_particles_list)  # shape: (T+1, N, 2)
    particle_paths = _reconstruct_particle_paths(position_history, parent_history)
    if path_selection == "sequence" or recorder.paths_enabled:
        sensor_headings = np.asarray(
            [heading.selected_heading for heading in step_headings],
            dtype=float,
        )
        turning_evidence = np.asarray(
            [
                (
                    heading.trajectory_movement_type == "turning"
                    or heading.movement_type == "turning"
                    or (
                        prepared_motion_posteriors is not None
                        and index < len(prepared_motion_posteriors)
                        and prepared_motion_posteriors[index].turning_probability
                        >= 0.25
                    )
                )
                for index, heading in enumerate(step_headings)
            ],
            dtype=bool,
        )
        sequence_path, sequence_modes, sequence_sources = _select_sequence_map_path(
            particle_paths,
            path_log_score_history[-1],
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
            sensor_headings,
            turning_evidence,
        )
        current_path, current_modes, current_sources = _select_reachable_cluster_path(
            position_history,
            weight_history,
            parent_history,
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        sequence_reversals = _unsupported_reversal_count(
            sequence_path,
            sensor_headings,
            turning_evidence,
        )
        current_reversals = _unsupported_reversal_count(
            current_path,
            sensor_headings,
            turning_evidence,
        )
        if path_selection == "sequence" and sequence_reversals < current_reversals:
            selected_path = sequence_path
            trajectory_modes = sequence_modes
            trajectory_sources = sequence_sources
            selected_mode = "sequence"
        else:
            selected_path = current_path
            trajectory_modes = current_modes
            trajectory_sources = current_sources
            selected_mode = "current"
    else:
        selected_path, trajectory_modes, trajectory_sources = (
            _select_reachable_cluster_path(
                position_history,
                weight_history,
                parent_history,
                map_gray,
                gx_mean,
                gz_mean,
                origin_px,
                scale,
            )
        )
        selected_mode = "current"
    if recorder.paths_enabled:
        recorder.paths.append(
            ParticlePathComparison(
                selected_mode=selected_mode,
                selected_path=selected_path.copy(),
                current_path=current_path.copy(),
                sequence_path=sequence_path.copy(),
                particle_paths=particle_paths.copy(),
                current_reversals=current_reversals,
                sequence_reversals=sequence_reversals,
            )
        )
    if recorder.diagnostics_enabled:
        for diagnostic_offset, (mode, source) in enumerate(
            zip(trajectory_modes[1:], trajectory_sources[1:], strict=True)
        ):
            collector_index = diagnostics_start_index + diagnostic_offset
            recorder.diagnostics[collector_index] = replace(
                recorder.diagnostics[collector_index],
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
