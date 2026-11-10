"""rikka の Click グループとコマンドを定義する。

役割:
    検証・入力解決・オプション群を組み合わせ、解析を起動する。
依存元:
    option_groups、validators、resolvers を使用し、commands を遅延 import する。
利用先:
    パッケージの main から CLI の入口として使用する。
処理フロー:
    Click が引数を受け取り、PDR・PF・描画の pipeline を呼ぶ実行処理へ渡す。
"""

from pathlib import Path

import click

from ..common.config import (
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_SCALE,
    PF_CANDIDATE_COUNT,
    TRAJECTORY_PLOT_MODE,
)
from .option_groups import (
    _ble_sample_options,
    _common_options,
    _evaluate_options,
    _landmark_path_options,
    _particle_options,
    _run_options,
    _sensor_options,
)
from .resolvers import _parse_landmark_specs as _parse_landmark_specs
from .resolvers import (
    _report_landmarks_without_passage,
    _resolve_ble_inputs,
    _resolve_free_origin_settings,
    _resolve_landmark_path_inputs,
    _resolve_landmark_path_order,
    _resolve_measurement_settings,
)
from .validators import (
    _validate_gyro_bias_options,
)


@click.group()
def cli() -> None:
    """rikka — PDR 歩行軌跡推定ツール"""


# ``rikka.cli`` サブパッケージの import 後も Click グループを参照できるよう保持する。
_click_cli = cli


def _run_pdr(
    ctx: click.Context,
    data_dir: str,
    floormap: str,
    origin_px: tuple[int, int],
    scale: float,
    direction: float,
    height_m: float,
    step_detection: str,
    step_length_method: str,
    heading_method: str,
    gyro_bias_method: str,
    gyro_bias: float | None,
    sidestep_lateral_ratio: float,
    sidestep_min_lateral_displacement: float,
    motion_heading_correction: str,
    sidestep_smoothing: str,
    forward_heading_source: str,
    sidestep_heading_source: str,
    sidestep_suspect_mode: str,
    motion_estimation: str,
    smoothing_mode: str,
    no_plot: bool,
    free_origin: bool,
    ble_landmark: bool,
    ble_detection_profile: str,
    ble_data_path: str,
    ble_rssi_threshold: float,
    ble_correction: str,
    ble_release_margin: float,
    ble_release_streak: int,
    ble_sync_window: float,
    ble_preflight: str,
    ble_max_correction: float,
    ble_max_warp_span: float | None,
    ble_retrofit_forward: str,
    ble_retrofit_max_heading: float,
    ble_retrofit_stride_scale_min: float,
    ble_retrofit_stride_scale_max: float,
    ble_retrofit_min_span: float,
    ble_retrofit_map_check: str,
    ble_retrofit_damp_factors: tuple[float, ...],
) -> None:
    from ..common.lib.sensors import load_sensor_data  # noqa: PLC0415
    from .commands import run as _run  # noqa: PLC0415

    _validate_gyro_bias_options(gyro_bias_method, gyro_bias)
    if free_origin:
        origin_px, direction, height_m = _resolve_free_origin_settings(
            ctx,
            data_dir,
            direction,
            height_m,
            ble_landmark,
        )
    else:
        origin_px, direction, height_m = _resolve_measurement_settings(
            ctx,
            data_dir,
            floormap,
            origin_px,
            direction,
            height_m,
        )
    df_acc, df_gyro = load_sensor_data(data_dir)
    ble_data_path, ble_landmarks = _resolve_ble_inputs(
        data_dir,
        ble_landmark,
        ble_data_path,
    )
    _run(
        df_acc=df_acc,
        df_gyro=df_gyro,
        plot=not no_plot,
        use_particle_filter=False,
        trajectory_plot="meter" if free_origin else TRAJECTORY_PLOT_MODE,
        floormap_path=floormap,
        origin_px=origin_px,
        scale=scale,
        initial_direction=direction,
        height_m=height_m,
        step_detection_method=step_detection,
        step_length_method=step_length_method,
        heading_method=heading_method,
        gyro_bias_method=gyro_bias_method,
        gyro_bias=gyro_bias,
        sidestep_lateral_ratio=sidestep_lateral_ratio,
        sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
        motion_heading_correction=motion_heading_correction,
        sidestep_smoothing=sidestep_smoothing,
        forward_heading_source=forward_heading_source,
        sidestep_heading_source=sidestep_heading_source,
        sidestep_suspect_mode=sidestep_suspect_mode,
        motion_estimation=motion_estimation,
        smoothing_mode=smoothing_mode,
        ble_landmark=ble_landmark,
        ble_detection_profile=ble_detection_profile,
        ble_data_path=ble_data_path,
        ble_rssi_threshold=ble_rssi_threshold,
        ble_correction=ble_correction,
        ble_release_margin=ble_release_margin,
        ble_release_streak=ble_release_streak,
        ble_sync_window=ble_sync_window,
        ble_preflight=ble_preflight,
        ble_max_correction=ble_max_correction,
        ble_max_warp_span=ble_max_warp_span,
        ble_retrofit_forward=ble_retrofit_forward,
        ble_retrofit_max_heading=ble_retrofit_max_heading,
        ble_retrofit_stride_scale_min=ble_retrofit_stride_scale_min,
        ble_retrofit_stride_scale_max=ble_retrofit_stride_scale_max,
        ble_retrofit_min_span=ble_retrofit_min_span,
        ble_retrofit_map_check=ble_retrofit_map_check,
        ble_retrofit_damp_factors=ble_retrofit_damp_factors,
        ble_landmarks=ble_landmarks,
    )


@cli.command()
@_common_options
@_run_options
@click.pass_context
def run(
    ctx: click.Context,
    data_dir: str,
    floormap: str,
    origin_px: tuple[int, int],
    scale: float,
    direction: float,
    height_m: float,
    step_detection: str,
    step_length_method: str,
    heading_method: str,
    gyro_bias_method: str,
    gyro_bias: float | None,
    sidestep_lateral_ratio: float,
    sidestep_min_lateral_displacement: float,
    motion_heading_correction: str,
    sidestep_smoothing: str,
    forward_heading_source: str,
    sidestep_heading_source: str,
    sidestep_suspect_mode: str,
    motion_estimation: str,
    smoothing_mode: str,
    no_plot: bool,
    free_origin: bool,
    ble_landmark: bool,
    ble_detection_profile: str,
    ble_data_path: str,
    ble_rssi_threshold: float,
    ble_correction: str,
    ble_release_margin: float,
    ble_release_streak: int,
    ble_sync_window: float,
    ble_preflight: str,
    ble_max_correction: float,
    ble_max_warp_span: float | None,
    ble_retrofit_forward: str,
    ble_retrofit_max_heading: float,
    ble_retrofit_stride_scale_min: float,
    ble_retrofit_stride_scale_max: float,
    ble_retrofit_min_span: float,
    ble_retrofit_map_check: str,
    ble_retrofit_damp_factors: tuple[float, ...],
) -> None:
    """決定論的 PDR で歩行軌跡を推定する。"""
    _run_pdr(
        ctx,
        data_dir,
        floormap,
        origin_px,
        scale,
        direction,
        height_m,
        step_detection,
        step_length_method,
        heading_method,
        gyro_bias_method,
        gyro_bias,
        sidestep_lateral_ratio,
        sidestep_min_lateral_displacement,
        motion_heading_correction,
        sidestep_smoothing,
        forward_heading_source,
        sidestep_heading_source,
        sidestep_suspect_mode,
        motion_estimation,
        smoothing_mode,
        no_plot,
        free_origin,
        ble_landmark,
        ble_detection_profile,
        ble_data_path,
        ble_rssi_threshold,
        ble_correction,
        ble_release_margin,
        ble_release_streak,
        ble_sync_window,
        ble_preflight,
        ble_max_correction,
        ble_max_warp_span,
        ble_retrofit_forward,
        ble_retrofit_max_heading,
        ble_retrofit_stride_scale_min,
        ble_retrofit_stride_scale_max,
        ble_retrofit_min_span,
        ble_retrofit_map_check,
        ble_retrofit_damp_factors,
    )


cli.add_command(run, name="pdr")


@cli.command()
@_particle_options
@click.pass_context
def particle(
    ctx: click.Context,
    data_dir: str,
    floormap: str,
    origin_px: tuple[int, int],
    scale: float,
    direction: float,
    height_m: float,
    step_detection: str,
    step_length_method: str,
    heading_method: str,
    gyro_bias_method: str,
    gyro_bias: float | None,
    sidestep_lateral_ratio: float,
    sidestep_min_lateral_displacement: float,
    motion_heading_correction: str,
    sidestep_smoothing: str,
    forward_heading_source: str,
    sidestep_heading_source: str,
    sidestep_suspect_mode: str,
    motion_estimation: str,
    smoothing_mode: str,
    no_plot: bool,
    ble_landmark: bool,
    ble_detection_profile: str,
    ble_data_path: str,
    ble_rssi_threshold: float,
    ble_correction: str,
    ble_release_margin: float,
    ble_release_streak: int,
    ble_sync_window: float,
    ble_preflight: str,
    ble_max_correction: float,
    ble_max_warp_span: float | None,
    ble_retrofit_forward: str,
    ble_retrofit_max_heading: float,
    ble_retrofit_stride_scale_min: float,
    ble_retrofit_stride_scale_max: float,
    ble_retrofit_min_span: float,
    ble_retrofit_map_check: str,
    ble_retrofit_damp_factors: tuple[float, ...],
    save_animation: bool,
    save_step_frames: bool,
    step_frames_range: tuple[int, int] | None,
    step_frames_arrows: int,
    step_frames_dpi: int,
    save_path_comparison: bool,
    history_candidate: bool,
    pf_particles: int,
    pf_init: str,
    pf_converge_cell: float,
    pf_converge_top_k: int,
    pf_converge_weight_ratio: float,
    pf_converge_spread: float,
    pf_converge_steps: int,
    pf_converge_restore_normal: bool,
    pf_seed: int | None,
    motion_predictive_weight_power: float,
    pf_path_selection: str,
    pf_landmark_mode: str,
    pf_landmark_retrofit: bool,
    pf_proposal: str,
    pf_candidates: int | None,
    pf_hard_anchors: Path | None,
    floormap_floor_id: str | None,
    pf_backward: bool = False,
) -> None:
    """オプションの互換性を検証し、PFとマップマッチングで歩行軌跡を推定する。"""
    from ..common.lib.sensors import load_sensor_data  # noqa: PLC0415
    from ..landmark.lib.hard_anchor import load_hard_positions  # noqa: PLC0415
    from .commands import run as _run  # noqa: PLC0415

    if pf_init == "uniform" and pf_hard_anchors is not None:
        raise click.UsageError(
            "--pf-init uniform では確定位置の経路選択が未対応のため、"
            "--pf-hard-anchors は併用できません。"
        )
    if (
        ble_landmark
        and pf_proposal == "candidate-bank"
        and pf_landmark_mode not in {"ranging", "observation", "none"}
    ):
        raise click.UsageError(
            "candidate-bank と --ble-landmark の併用には "
            "--pf-landmark-mode ranging（推奨）または observation を指定するか、"
            "--pf-proposal legacy または no-relocation を指定してください。"
        )
    hard_positions = (
        () if pf_hard_anchors is None else load_hard_positions(pf_hard_anchors)
    )
    _validate_gyro_bias_options(gyro_bias_method, gyro_bias)
    if pf_candidates is None:
        pf_candidates = PF_CANDIDATE_COUNT if pf_proposal == "candidate-bank" else 1
    if pf_proposal != "candidate-bank" and pf_candidates != 1:
        raise click.BadParameter(
            "1以外の候補数には --pf-proposal candidate-bank が必要です。",
            param_hint="--pf-candidates",
        )
    if pf_init == "uniform":
        origin_px, direction, height_m = _resolve_free_origin_settings(
            ctx,
            data_dir,
            direction,
            height_m,
            ble_landmark,
            option_label="--pf-init uniform",
            allow_ble_landmark=True,
        )
    else:
        origin_px, direction, height_m = _resolve_measurement_settings(
            ctx,
            data_dir,
            floormap,
            origin_px,
            direction,
            height_m,
        )
    df_acc, df_gyro = load_sensor_data(data_dir)
    ble_data_path, ble_landmarks = _resolve_ble_inputs(
        data_dir,
        ble_landmark,
        ble_data_path,
    )
    _run(
        df_acc=df_acc,
        df_gyro=df_gyro,
        plot=not no_plot,
        save_animation=True if save_animation else None,
        save_step_frames=save_step_frames,
        step_frames_range=step_frames_range,
        step_frames_arrows=step_frames_arrows,
        step_frames_dpi=step_frames_dpi,
        save_path_comparison=save_path_comparison,
        history_candidate=history_candidate,
        use_particle_filter=True,
        floormap_path=floormap,
        origin_px=origin_px,
        scale=scale,
        initial_direction=direction,
        height_m=height_m,
        step_detection_method=step_detection,
        step_length_method=step_length_method,
        heading_method=heading_method,
        gyro_bias_method=gyro_bias_method,
        gyro_bias=gyro_bias,
        sidestep_lateral_ratio=sidestep_lateral_ratio,
        sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
        motion_heading_correction=motion_heading_correction,
        sidestep_smoothing=sidestep_smoothing,
        forward_heading_source=forward_heading_source,
        sidestep_heading_source=sidestep_heading_source,
        sidestep_suspect_mode=sidestep_suspect_mode,
        motion_estimation=motion_estimation,
        smoothing_mode=smoothing_mode,
        particle_seed=pf_seed,
        particle_count=pf_particles,
        init_mode=pf_init,
        pf_backward=pf_backward,
        convergence_cell_m=pf_converge_cell,
        convergence_top_k=pf_converge_top_k,
        convergence_weight_ratio=pf_converge_weight_ratio,
        convergence_spread_m=pf_converge_spread,
        convergence_single_steps=pf_converge_steps,
        convergence_restore_normal=pf_converge_restore_normal,
        motion_predictive_weight_power=motion_predictive_weight_power,
        pf_path_selection=pf_path_selection,
        pf_landmark_mode=pf_landmark_mode,
        pf_landmark_retrofit=pf_landmark_retrofit,
        hard_positions=hard_positions,
        floormap_floor_id=floormap_floor_id,
        pf_proposal=pf_proposal,
        pf_candidates=pf_candidates,
        ble_landmark=ble_landmark,
        ble_detection_profile=ble_detection_profile,
        ble_data_path=ble_data_path,
        ble_rssi_threshold=ble_rssi_threshold,
        ble_correction=ble_correction,
        ble_release_margin=ble_release_margin,
        ble_release_streak=ble_release_streak,
        ble_sync_window=ble_sync_window,
        ble_preflight=ble_preflight,
        ble_max_correction=ble_max_correction,
        ble_max_warp_span=ble_max_warp_span,
        ble_retrofit_forward=ble_retrofit_forward,
        ble_retrofit_max_heading=ble_retrofit_max_heading,
        ble_retrofit_stride_scale_min=ble_retrofit_stride_scale_min,
        ble_retrofit_stride_scale_max=ble_retrofit_stride_scale_max,
        ble_retrofit_min_span=ble_retrofit_min_span,
        ble_retrofit_map_check=ble_retrofit_map_check,
        ble_retrofit_damp_factors=ble_retrofit_damp_factors,
        ble_landmarks=ble_landmarks,
    )


@cli.command()
@_sensor_options
def sensor(
    data_dir: str,
    step_detection: str,
    gyro_bias_method: str,
    gyro_bias: float | None,
) -> None:
    """センサーデータをグラフ化して入力フォルダに保存する。"""
    from ..plot.pipeline import render_sensor  # noqa: PLC0415

    _validate_gyro_bias_options(gyro_bias_method, gyro_bias)
    render_sensor(
        data_dir,
        step_detection_method=step_detection,
        gyro_bias_method=gyro_bias_method,
        gyro_bias=gyro_bias,
    )


@cli.command(name="ble-sample")
@_ble_sample_options
def ble_sample(
    mode: str,
    source: str,
    truth_csv: str,
    data_dir: str,
    output: str,
    seed: int,
) -> None:
    """歩行データと同じ時間軸のサンプル BLE RSSI CSV を生成する。"""
    import numpy as np  # noqa: PLC0415

    from ..ble.lib.sample import (  # noqa: PLC0415
        generate_sample_csv,
        load_truth_trajectory,
        map_truth_to_step_times,
    )
    from ..common.lib.floormap import compute_meter_coords  # noqa: PLC0415
    from ..common.lib.sensors import load_sensor_data  # noqa: PLC0415
    from ..common.settings import (  # noqa: PLC0415
        BleLandmarkSettings,
        BleSampleSettings,
        PdrSettings,
    )
    from ..pdr.pipeline import run_pdr  # noqa: PLC0415

    df_acc, df_gyro = load_sensor_data(data_dir)
    pdr_result = run_pdr(PdrSettings(), df_acc, df_gyro)
    trajectory: list[list[float]] | np.ndarray = pdr_result.trajectory
    trajectory_times: list[float] | np.ndarray = pdr_result.t_at_steps
    if mode == "distance" and source == "truth":
        truth_xy = load_truth_trajectory(truth_csv)
        mapped, mapped_times = map_truth_to_step_times(
            truth_xy,
            pdr_result.t_at_steps,
            pdr_result.trajectory,
        )
        trajectory = mapped
        trajectory_times = mapped_times[1:]
    landmarks = BleLandmarkSettings().landmarks
    meter_xs, meter_ys = compute_meter_coords(
        np.asarray([item.pixel_x for item in landmarks]),
        np.asarray([item.pixel_y for item in landmarks]),
        pdr_result.prepared.gx_mean,
        pdr_result.prepared.gz_mean,
        FLOORMAP_ORIGIN_PX,
        FLOORMAP_SCALE,
    )
    landmark_positions = {
        item.beacon_id: (float(x), float(y))
        for item, x, y in zip(landmarks, meter_xs, meter_ys, strict=True)
    }
    path, rows = generate_sample_csv(
        df_acc,
        output,
        BleSampleSettings(mode=mode, seed=seed),
        trajectory=trajectory,
        t_at_steps=trajectory_times,
        landmark_positions=landmark_positions,
    )
    print(f"Sample BLE RSSI saved to {path} ({rows} rows)")


@cli.command(name="landmark-path")
@_landmark_path_options
def landmark_path(
    data_dir: str,
    landmark_specs: tuple[str, ...],
    order: str,
    floormap: str,
    scale: float,
    no_plot: bool,
) -> None:
    """ランドマーク座標のみを順に直線で結んだ軌跡を出力する。"""
    from .commands import run_landmark_path  # noqa: PLC0415

    landmarks, source, ble_data_path = _resolve_landmark_path_inputs(
        data_dir,
        landmark_specs,
    )
    if len(landmarks) < 2:
        raise click.UsageError(
            f"経路には2点以上のランドマークが必要です（出所: {source}）。"
        )
    order_times = (
        None
        if order == "given"
        else _resolve_landmark_path_order(ble_data_path, landmarks)
    )
    click.echo(f"ランドマーク座標の出所: {source}")
    click.echo(
        "経路順: "
        + ("計測BLEの平滑RSSIピーク時刻" if order_times is not None else "座標の指定順")
    )
    if order_times is not None:
        _report_landmarks_without_passage(landmarks, order_times, source)
    output_dir = run_landmark_path(
        landmarks,
        floormap_path=floormap,
        scale=scale,
        plot=not no_plot,
        order_times=order_times,
    )
    click.echo(f"ランドマーク経路の出力先: {output_dir}")


@cli.command(name="evaluate")
@_evaluate_options
def evaluate(
    estimated_csv: Path,
    truth_csv: Path,
    coordinate_system: str,
    floor_scale: float,
    tolerance_m: float,
    output_dir: Path,
) -> None:
    """推定軌跡CSVを正解軌跡CSVと比較する。"""
    from ..common.lib.trajectory_evaluation import (  # noqa: PLC0415
        evaluate_trajectory_csv,
        write_evaluation_outputs,
    )

    evaluation, summary = evaluate_trajectory_csv(
        estimated_csv,
        truth_csv,
        coordinate_system=coordinate_system,
        floor_scale=floor_scale,
        tolerance_m=tolerance_m,
    )
    csv_path, json_path = write_evaluation_outputs(
        evaluation,
        summary,
        output_dir,
    )
    click.echo(f"評価CSV: {csv_path}")
    click.echo(f"集計JSON: {json_path}")
    click.echo(f"RMSE: {summary.rmse_m:.3f} m")
    click.echo(f"MAE: {summary.mae_m:.3f} m")
    click.echo(f"評価対象率: {summary.point_coverage_rate:.1f}%")


def main() -> None:
    cli()
