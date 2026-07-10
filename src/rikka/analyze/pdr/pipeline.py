"""PDR CLI パイプライン。

CLI や外部呼び出しから見た PDR 実行の入口。センサー読み込み、共通ステップ準備、
通常 PDR / particle filter の分岐、CSV・プロット出力をここでまとめる。
アルゴリズム本体は各モジュールへ分離し、このファイルは orchestration に集中する。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ...config import (
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    INITIAL_DIRECTION,
    SAMPLING_RATE,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    USER_HEIGHT_M,
)
from .common import (
    _validate_forward_heading_source,
    _validate_motion_heading_correction,
    _validate_non_negative_parameter,
    _validate_positive_parameter,
    _validate_scale,
    _validate_sidestep_heading_source,
    _validate_sidestep_smoothing,
    _validate_sidestep_suspect_mode,
)
from .models import GyroBiasResult
from .outputs import (
    _build_gyro_bias_dataframe,
    _build_step_headings_dataframe,
    _build_step_segments_dataframe,
    _build_step_vectors_dataframe,
    _build_trajectory_dataframe,
    _create_output_dir,
    _step_plot_signal,
)
from .plotting import plot_trajectory
from .sensors import load_sensor_data
from .trajectory import prepare_pdr_steps


def run(
    df_acc: pd.DataFrame | None = None,
    df_gyro: pd.DataFrame | None = None,
    plot: bool = True,
    use_particle_filter: bool = False,
    save_animation: bool | None = None,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    initial_direction: float = INITIAL_DIRECTION,
    height_m: float = USER_HEIGHT_M,
    step_detection_method: str | None = None,
    heading_method: str | None = None,
    gyro_bias_method: str | None = None,
    gyro_bias: float | None = None,
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO,
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    motion_heading_correction: str = "auto",
    sidestep_smoothing: str = SIDESTEP_SMOOTHING_METHOD,
    forward_heading_source: str = FORWARD_HEADING_SOURCE,
    sidestep_heading_source: str = "motion",
    sidestep_suspect_mode: str = "motion",
) -> pd.DataFrame:
    """PDRのメインパイプラインを実行する。

    センサーデータの読み込みから軌跡の推定・CSV保存・表示までを一括して実行する。
    処理の流れ: データ読み込み → センサー処理 → ステップ検出 → 軌跡推定 → CSV保存 → 表示

    Args:
        df_acc (pd.DataFrame | None):
            加速度データ（列: t, x, y, z）。省略時は ``DATA_DIR`` の CSV から読み込む。
            渡す場合は ``load_sensor_data()`` によるリネーム後の列名
            （t, x, y, z）を使うこと。
        df_gyro (pd.DataFrame | None):
            ジャイロスコープデータ（列: t, x, y, z）。
            省略時は ``DATA_DIR`` の CSV から読み込む。
            ``df_acc`` と必ずセットで渡すこと。
        plot (bool):
            ``True`` のとき軌跡をプロット表示する。
            バッチ処理やCI環境では ``False`` を指定する。デフォルトは ``True``。
        use_particle_filter (bool):
            ``True`` のときパーティクルフィルタで軌跡を推定する。
            デフォルトは ``False``。
        save_animation (bool | None):
            パーティクルフィルタのアニメーション保存を制御する。
            ``None`` のときは ``plot`` と同じ値を使う。
        floormap_path (str | Path):
            フロアマップ画像のパス。デフォルトは ``FLOORMAP_PATH``。
        origin_px (tuple[int, int]):
            軌跡起点のピクセル座標 ``(x, y)``。デフォルトは ``FLOORMAP_ORIGIN_PX``。
        scale (float):
            1ピクセルあたりのメートル数。デフォルトは ``FLOORMAP_SCALE``。
        initial_direction (float):
            歩行開始方向のオフセット [度]。デフォルトは ``INITIAL_DIRECTION``。
        height_m (float):
            Weinbergモデルのスケール係数を補正するユーザー身長 [m]。
        step_detection_method:
            ステップ検出手法。``None`` のときは設定値を使用する。
        heading_method:
            方位推定手法。``None`` のときは設定値を使用する。
        gyro_bias_method:
            ジャイロバイアス推定手法。``None`` のときは設定値を使用する。
        gyro_bias:
            ``gyro_bias_method="manual"`` のときに使う手動バイアス [rad/s]。
        sidestep_lateral_ratio:
            横歩き判定に使う横方向/前方向の最小比率。
        sidestep_min_lateral_displacement:
            横歩き判定に必要な横方向変位の最小値 [m]。
        motion_heading_correction:
            水平加速度移動方向の固定ずれ補正モード（``"auto"`` or ``"none"``）。
        sidestep_smoothing:
            横歩き判定の平滑化モード
            （``"none"``、``"isolated"``、``"clustered"``）。
        forward_heading_source:
            forward 判定ステップの軌跡方位ソース（``"body"`` or ``"motion"``）。
        sidestep_heading_source:
            確定横歩きステップの軌跡方位ソース。
        sidestep_suspect_mode:
            横歩き疑いステップの軌跡反映モード。
    Returns:
        pd.DataFrame: 軌跡データ（列: timestamp_s, x, y）

    Raises:
        ValueError: ``df_acc`` と ``df_gyro`` の片方だけが渡された場合
    """
    _validate_scale(scale)
    sidestep_lateral_ratio = _validate_positive_parameter(
        "sidestep_lateral_ratio",
        sidestep_lateral_ratio,
    )
    sidestep_min_lateral_displacement = _validate_non_negative_parameter(
        "sidestep_min_lateral_displacement",
        sidestep_min_lateral_displacement,
    )
    selected_motion_heading_correction = _validate_motion_heading_correction(
        motion_heading_correction
    )
    selected_sidestep_smoothing = _validate_sidestep_smoothing(sidestep_smoothing)
    selected_forward_heading_source = _validate_forward_heading_source(
        forward_heading_source
    )
    selected_sidestep_heading_source = _validate_sidestep_heading_source(
        sidestep_heading_source
    )
    selected_sidestep_suspect_mode = _validate_sidestep_suspect_mode(
        sidestep_suspect_mode
    )
    output_dir = _create_output_dir()
    should_save_animation = plot if save_animation is None else save_animation

    if (df_acc is None) != (df_gyro is None):
        raise ValueError("df_acc と df_gyro は両方渡すか、両方省略してください。")

    if df_acc is None and df_gyro is None:
        df_acc, df_gyro = load_sensor_data()

    if df_acc is None or df_gyro is None:
        raise RuntimeError("内部エラー: df_acc または df_gyro が None（到達不能）")

    # 通常 PDR と particle filter で共有する決定論的ステップ情報を先に作る。
    prepared_steps = prepare_pdr_steps(
        df_acc,
        df_gyro,
        initial_direction=initial_direction,
        height_m=height_m,
        step_detection_method=step_detection_method,
        heading_method=heading_method,
        gyro_bias_method=gyro_bias_method,
        gyro_bias=gyro_bias,
        sidestep_lateral_ratio=sidestep_lateral_ratio,
        sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
        motion_heading_correction=selected_motion_heading_correction,
        sidestep_smoothing=selected_sidestep_smoothing,
        forward_heading_source=selected_forward_heading_source,
        sidestep_heading_source=selected_sidestep_heading_source,
        sidestep_suspect_mode=selected_sidestep_suspect_mode,
    )
    df_acc = prepared_steps.df_acc
    df_gyro = prepared_steps.df_gyro
    step_detection = prepared_steps.step_detection
    peaks = step_detection.peaks
    weinberg_k = prepared_steps.weinberg_k
    selected_heading_method = prepared_steps.heading_method
    print(f"Weinberg K: {weinberg_k:.3f} (height={height_m:.2f} m)")
    print(f"Heading method: {selected_heading_method}")
    print(
        "Sidestep detection: "
        f"ratio={sidestep_lateral_ratio:.3f} "
        f"min_lateral={sidestep_min_lateral_displacement:.3f} m "
        f"motion_heading_correction={selected_motion_heading_correction} "
        f"smoothing={selected_sidestep_smoothing} "
        f"forward_heading_source={selected_forward_heading_source} "
        f"sidestep_heading_source={selected_sidestep_heading_source} "
        f"sidestep_suspect_mode={selected_sidestep_suspect_mode}"
    )
    bias_result = df_gyro.attrs.get("gyro_bias_result")
    if isinstance(bias_result, GyroBiasResult):
        if bias_result.calibration_start_s is None:
            window_text = "manual"
        else:
            window_text = (
                f"{bias_result.calibration_start_s:.3f}-"
                f"{bias_result.calibration_end_s:.3f}s"
            )
        print(
            "Gyro bias: "
            f"method={bias_result.method} "
            f"bias={bias_result.bias_rad_s:.6f} rad/s "
            f"window={window_text} "
            f"fallback={bias_result.fallback_reason}"
        )
    if step_detection.threshold is None:
        print(f"Step detection: {step_detection.method}")
    else:
        print(
            "Step detection: "
            f"{step_detection.method} "
            f"threshold={step_detection.threshold:.3f} "
            f"polarity={step_detection.polarity}"
        )

    # 重力成分の平均を算出（Y軸反転の自動判定に使用）
    gx_mean = prepared_steps.gx_mean
    gz_mean = prepared_steps.gz_mean
    dominant = "X軸" if abs(gx_mean) > abs(gz_mean) else "Z軸"
    y_flipped = (abs(gx_mean) > abs(gz_mean) and gx_mean > 0) or (
        abs(gz_mean) >= abs(gx_mean) and gz_mean < 0
    )
    print(
        f"重力主成分: {dominant}  gx={gx_mean:.2f}, gz={gz_mean:.2f} m/s²"
        f" → Y軸{'反転' if y_flipped else '非反転'}"
    )

    df_gyro_bias = _build_gyro_bias_dataframe(df_gyro)
    gyro_bias_path = output_dir / "gyro_bias.csv"
    df_gyro_bias.to_csv(gyro_bias_path, index=False)
    print(f"Gyro bias saved to {gyro_bias_path}")

    # particle filter は prepared_steps を受け取り、同じステップ列を地図制約で補正する。
    if use_particle_filter:
        from ..particle_filter import (  # noqa: PLC0415
            plot_particle_filter_trajectory,
            run_particle_filter,
            save_particle_animation,
        )

        (
            trajectory,
            step_lengths,
            t_at_steps,
            all_particles,
            step_headings,
        ) = run_particle_filter(
            peaks,
            df_gyro,
            df_acc,
            gx_mean,
            gz_mean,
            floormap_path=floormap_path,
            origin_px=origin_px,
            scale=scale,
            initial_direction=initial_direction,
            weinberg_k=weinberg_k,
            heading_method=selected_heading_method,
            step_segments=step_detection.segments,
            prepared_step_headings=prepared_steps.step_headings,
            prepared_step_lengths=prepared_steps.step_lengths,
            prepared_step_times=prepared_steps.t_at_steps,
            sidestep_lateral_ratio=sidestep_lateral_ratio,
            sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
            motion_heading_correction=selected_motion_heading_correction,
            sidestep_smoothing=selected_sidestep_smoothing,
            forward_heading_source=selected_forward_heading_source,
            sidestep_heading_source=selected_sidestep_heading_source,
            sidestep_suspect_mode=selected_sidestep_suspect_mode,
        )

        print(f"Peaks detected: {len(peaks)}")
        print(f"Steps used: {len(step_lengths)}")
        for i, (x, y) in enumerate(trajectory):
            print(f"step {i}: ({x:.3f}, {y:.3f})")

        df_trajectory = _build_trajectory_dataframe(
            trajectory,
            t_at_steps,
        )
        output_path = output_dir / "trajectory.csv"
        df_trajectory.to_csv(output_path, index=False)
        print(f"Trajectory saved to {output_path}")

        df_step_lengths = pd.DataFrame(
            {"step": range(1, len(step_lengths) + 1), "step_length_m": step_lengths}
        )
        step_length_path = output_dir / "step_lengths.csv"
        df_step_lengths.to_csv(step_length_path, index=False)
        print(f"Step lengths saved to {step_length_path}")

        df_step_vectors = _build_step_vectors_dataframe(trajectory)
        step_vector_path = output_dir / "step_vectors.csv"
        df_step_vectors.to_csv(step_vector_path, index=False)
        print(f"Step vectors saved to {step_vector_path}")

        df_step_headings = _build_step_headings_dataframe(step_headings)
        step_heading_path = output_dir / "step_headings.csv"
        df_step_headings.to_csv(step_heading_path, index=False)
        print(f"Step headings saved to {step_heading_path}")

        if step_detection.method == "paper_vertical_threshold":
            df_step_segments = _build_step_segments_dataframe(
                df_acc,
                step_detection.segments,
            )
            step_segment_path = output_dir / "step_segments.csv"
            df_step_segments.to_csv(step_segment_path, index=False)
            print(f"Step segments saved to {step_segment_path}")

        if plot:
            plot_particle_filter_trajectory(
                trajectory,
                gx_mean=gx_mean,
                gz_mean=gz_mean,
                floormap_path=floormap_path,
                origin_px=origin_px,
                scale=scale,
                output_dir=output_dir,
                step_headings=step_headings,
            )
            from ..sensor_plot import (  # noqa: PLC0415
                plot_step_lengths,
                plot_step_vectors,
            )

            t_acc = (
                df_acc["t"].to_numpy()
                if "t" in df_acc.columns
                else np.arange(len(df_acc)) / SAMPLING_RATE
            )
            step_signal, step_signal_label, step_signal_threshold = _step_plot_signal(
                df_acc,
                step_detection,
            )
            plot_step_lengths(
                step_lengths,
                output_dir,
                t_at_steps=t_at_steps,
                t_acc=t_acc,
                step_signal=step_signal,
                step_signal_label=step_signal_label,
                step_signal_threshold=step_signal_threshold,
            )
            plot_step_vectors(
                trajectory,
                output_dir,
                df_acc=df_acc,
                df_gyro=df_gyro,
                peaks=peaks,
                step_headings=step_headings,
                initial_direction=initial_direction,
            )

        if should_save_animation:
            save_particle_animation(
                all_particles,
                trajectory,
                gx_mean=gx_mean,
                gz_mean=gz_mean,
                floormap_path=floormap_path,
                origin_px=origin_px,
                scale=scale,
                output_path=output_dir / "particle_filter.mp4",
            )
    else:
        # 通常 PDR は prepared_steps の軌跡をそのまま出力する。
        trajectory = prepared_steps.trajectory
        step_lengths = prepared_steps.step_lengths
        t_at_steps = prepared_steps.t_at_steps
        step_headings = prepared_steps.step_headings

        print(f"Peaks detected: {len(peaks)}")
        print(f"Steps used: {len(step_lengths)}")
        for i, (x, y) in enumerate(trajectory):
            print(f"step {i}: ({x:.3f}, {y:.3f})")

        df_trajectory = _build_trajectory_dataframe(
            trajectory,
            t_at_steps,
        )

        # 軌跡データをoutputフォルダにCSVとして保存
        output_path = output_dir / "trajectory.csv"
        df_trajectory.to_csv(output_path, index=False)
        print(f"Trajectory saved to {output_path}")

        # 歩幅データをoutputフォルダにCSVとして保存
        df_step_lengths = pd.DataFrame(
            {"step": range(1, len(step_lengths) + 1), "step_length_m": step_lengths}
        )
        step_length_path = output_dir / "step_lengths.csv"
        df_step_lengths.to_csv(step_length_path, index=False)
        print(f"Step lengths saved to {step_length_path}")

        df_step_vectors = _build_step_vectors_dataframe(trajectory)
        step_vector_path = output_dir / "step_vectors.csv"
        df_step_vectors.to_csv(step_vector_path, index=False)
        print(f"Step vectors saved to {step_vector_path}")

        df_step_headings = _build_step_headings_dataframe(step_headings)
        step_heading_path = output_dir / "step_headings.csv"
        df_step_headings.to_csv(step_heading_path, index=False)
        print(f"Step headings saved to {step_heading_path}")

        if step_detection.method == "paper_vertical_threshold":
            df_step_segments = _build_step_segments_dataframe(
                df_acc,
                step_detection.segments,
            )
            step_segment_path = output_dir / "step_segments.csv"
            df_step_segments.to_csv(step_segment_path, index=False)
            print(f"Step segments saved to {step_segment_path}")

        if plot:
            plot_trajectory(
                trajectory,
                gx_mean=gx_mean,
                gz_mean=gz_mean,
                floormap_path=floormap_path,
                origin_px=origin_px,
                scale=scale,
                output_dir=output_dir,
                step_headings=step_headings,
            )
            from ..sensor_plot import (  # noqa: PLC0415
                plot_step_lengths,
                plot_step_vectors,
            )

            t_acc = (
                df_acc["t"].to_numpy()
                if "t" in df_acc.columns
                else np.arange(len(df_acc)) / SAMPLING_RATE
            )
            step_signal, step_signal_label, step_signal_threshold = _step_plot_signal(
                df_acc,
                step_detection,
            )
            plot_step_lengths(
                step_lengths,
                output_dir,
                t_at_steps=t_at_steps,
                t_acc=t_acc,
                step_signal=step_signal,
                step_signal_label=step_signal_label,
                step_signal_threshold=step_signal_threshold,
            )
            plot_step_vectors(
                trajectory,
                output_dir,
                df_acc=df_acc,
                df_gyro=df_gyro,
                peaks=peaks,
                step_headings=step_headings,
                initial_direction=initial_direction,
            )

    return df_trajectory
