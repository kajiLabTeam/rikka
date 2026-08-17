"""rikka の Click オプションとコマンド定義。

役割:
    Click で ``run`` / ``pdr`` / ``particle`` / ``sensor`` コマンドと共通オプションを
    定義し、ライブラリ内部の実行関数へ利用者の指定を渡す。
依存元:
    ``config`` から CLI の既定値、``matplotlib_config`` からキャッシュ設定、
    ``ping`` から接続確認関数を取得する。実行時には循環 import を避けるため、
    ``analyze.pdr`` または ``analyze.sensor_plot`` を遅延 import する。
利用先:
    ``pyproject.toml`` の ``rikka = "rikka:main"`` から起動されるほか、
    パッケージ利用者へ ``ping`` を公開する。
処理フロー:
    起動時に Matplotlib を設定し、Click が引数を検証した後、通常 PDR または
    particle filter の pipeline、もしくはセンサー描画処理を呼び出す。
"""

from math import isfinite

import click

from ..common.config import (
    BLE_DATA_PATH,
    BLE_LANDMARK_ENABLED,
    BLE_RSSI_THRESHOLD_DBM,
    BLE_SAMPLE_MODE,
    BLE_SAMPLE_SEED,
    BLE_SAMPLE_TRUTH_PATH,
    DATA_DIR,
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    GYRO_BIAS_METHOD,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    MOTION_ESTIMATION,
    PF_LANDMARK_MODE,
    PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    PF_NUM_PARTICLES,
    PF_PATH_SELECTION,
    PF_STEP_FRAMES_ARROWS,
    PF_STEP_FRAMES_DPI,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    SIDESTEP_SUSPECT_MODE,
    SMOOTHING_MODE,
    STEP_DETECTION_METHOD,
    STEP_LENGTH_METHOD,
    USER_HEIGHT_M,
)
from ..common.lib.validation import (
    BLE_SAMPLE_MODES,
    BLE_SAMPLE_SOURCES,
    FORWARD_HEADING_SOURCES,
    GYRO_BIAS_METHODS,
    HEADING_METHODS,
    MOTION_ESTIMATION_METHODS,
    MOTION_HEADING_CORRECTION_METHODS,
    PF_LANDMARK_MODES,
    PF_PATH_SELECTION_METHODS,
    SIDESTEP_HEADING_SOURCES,
    SIDESTEP_SMOOTHING_METHODS,
    SIDESTEP_SUSPECT_MODES,
    SMOOTHING_MODES,
    STEP_DETECTION_METHODS,
    STEP_LENGTH_METHODS,
)

_DATA_DIR_DEFAULT = DATA_DIR
_BLE_DATA_DEFAULT = BLE_DATA_PATH
_BLE_LANDMARK_DEFAULT = BLE_LANDMARK_ENABLED
_BLE_RSSI_THRESHOLD_DEFAULT = BLE_RSSI_THRESHOLD_DBM
_BLE_SAMPLE_SEED_DEFAULT = BLE_SAMPLE_SEED
_BLE_SAMPLE_MODE_DEFAULT = BLE_SAMPLE_MODE
_BLE_SAMPLE_TRUTH_DEFAULT = BLE_SAMPLE_TRUTH_PATH
_FLOORMAP_DEFAULT = FLOORMAP_PATH
_ORIGIN_DEFAULT = FLOORMAP_ORIGIN_PX
_SCALE_DEFAULT = FLOORMAP_SCALE
_DIRECTION_DEFAULT = INITIAL_DIRECTION
_HEIGHT_DEFAULT = USER_HEIGHT_M
_STEP_DETECTION_DEFAULT = STEP_DETECTION_METHOD
_STEP_DETECTION_CHOICES = STEP_DETECTION_METHODS
_STEP_LENGTH_DEFAULT = STEP_LENGTH_METHOD
_STEP_LENGTH_CHOICES = STEP_LENGTH_METHODS
_HEADING_METHOD_DEFAULT = HEADING_METHOD
_HEADING_METHOD_CHOICES = HEADING_METHODS
_GYRO_BIAS_METHOD_DEFAULT = GYRO_BIAS_METHOD
_GYRO_BIAS_METHOD_CHOICES = GYRO_BIAS_METHODS
_SIDESTEP_LATERAL_RATIO_DEFAULT = SIDESTEP_LATERAL_RATIO
_SIDESTEP_MIN_LATERAL_DISPLACEMENT_DEFAULT = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M
_MOTION_HEADING_CORRECTION_DEFAULT = "auto"
_MOTION_HEADING_CORRECTION_CHOICES = MOTION_HEADING_CORRECTION_METHODS
_SIDESTEP_SMOOTHING_DEFAULT = SIDESTEP_SMOOTHING_METHOD
_SIDESTEP_SMOOTHING_CHOICES = SIDESTEP_SMOOTHING_METHODS
_FORWARD_HEADING_SOURCE_DEFAULT = FORWARD_HEADING_SOURCE
_FORWARD_HEADING_SOURCE_CHOICES = FORWARD_HEADING_SOURCES
_SIDESTEP_HEADING_SOURCE_DEFAULT = "motion"
_SIDESTEP_HEADING_SOURCE_CHOICES = SIDESTEP_HEADING_SOURCES
_SIDESTEP_SUSPECT_MODE_DEFAULT = SIDESTEP_SUSPECT_MODE
_SIDESTEP_SUSPECT_MODE_CHOICES = SIDESTEP_SUSPECT_MODES
_MOTION_ESTIMATION_DEFAULT = MOTION_ESTIMATION
_MOTION_ESTIMATION_CHOICES = MOTION_ESTIMATION_METHODS
_SMOOTHING_MODE_DEFAULT = SMOOTHING_MODE
_SMOOTHING_MODE_CHOICES = SMOOTHING_MODES
_PF_MOTION_PREDICTIVE_WEIGHT_POWER_DEFAULT = PF_MOTION_PREDICTIVE_WEIGHT_POWER
_PF_LANDMARK_MODE_DEFAULT = PF_LANDMARK_MODE
_PF_LANDMARK_MODE_CHOICES = PF_LANDMARK_MODES
_PF_NUM_PARTICLES_DEFAULT = PF_NUM_PARTICLES
_PF_PATH_SELECTION_DEFAULT = PF_PATH_SELECTION
_PF_PATH_SELECTION_CHOICES = PF_PATH_SELECTION_METHODS
_PF_STEP_FRAMES_ARROWS_DEFAULT = PF_STEP_FRAMES_ARROWS
_PF_STEP_FRAMES_DPI_DEFAULT = PF_STEP_FRAMES_DPI


def _validate_cli_scale(
    _ctx: click.Context,
    _param: click.Parameter,
    value: float,
) -> float:
    """scale が正の値であることを確認する。"""
    if not isfinite(value) or value <= 0:
        raise click.BadParameter("scale は有限な正の値を指定してください。")
    return value


def _validate_cli_positive_float(
    _ctx: click.Context,
    param: click.Parameter,
    value: float,
) -> float:
    """正の float オプションであることを確認する。"""
    if not isfinite(value) or value <= 0:
        raise click.BadParameter(f"{param.name} は有限な正の値を指定してください。")
    return value


def _validate_cli_non_negative_float(
    _ctx: click.Context,
    param: click.Parameter,
    value: float,
) -> float:
    """0以上の float オプションであることを確認する。"""
    if not isfinite(value) or value < 0:
        raise click.BadParameter(f"{param.name} は有限な0以上の値を指定してください。")
    return value


def _validate_cli_step_frames_range(
    _ctx: click.Context,
    _param: click.Parameter,
    value: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """歩画像の範囲が1始まりで昇順であることを確認する。"""
    if value is None:
        return None
    first, last = value
    if first < 1 or first > last:
        raise click.BadParameter("1 <= A <= B を満たす範囲を指定してください。")
    return value


def _validate_cli_finite_float(
    _ctx: click.Context,
    param: click.Parameter,
    value: float | None,
) -> float | None:
    """任意符号の float オプションが有限であることを確認する。"""
    if value is None:
        return None
    if not isfinite(value):
        raise click.BadParameter(f"{param.name} は有限な値を指定してください。")
    return value


def _validate_gyro_bias_options(
    gyro_bias_method: str,
    gyro_bias: float | None,
) -> None:
    """manual 指定時は明示的なジャイロバイアス値を必須にする。"""
    if gyro_bias_method == "manual" and gyro_bias is None:
        raise click.UsageError(
            "--gyro-bias-method manual を使う場合は --gyro-bias を指定してください。"
        )
    if gyro_bias is not None and not isfinite(gyro_bias):
        raise click.BadParameter("gyro_bias は有限な値を指定してください。")


def _common_options(f: click.decorators.FC) -> click.decorators.FC:
    """run / particle コマンド共通オプションをまとめたデコレータ。"""
    f = click.option(
        "--smoothing",
        "smoothing_mode",
        type=click.Choice(_SMOOTHING_MODE_CHOICES),
        default=_SMOOTHING_MODE_DEFAULT,
        show_default=True,
        help="適応推定の因果処理またはオフライン平滑化",
    )(f)
    f = click.option(
        "--motion-estimation",
        type=click.Choice(_MOTION_ESTIMATION_CHOICES),
        default=_MOTION_ESTIMATION_DEFAULT,
        show_default=True,
        help="運動状態・方位・歩幅の推定方式",
    )(f)
    f = click.option(
        "--no-plot", is_flag=True, default=False, help="グラフ表示を無効化"
    )(f)
    f = click.option(
        "--ble-rssi-threshold",
        type=float,
        default=_BLE_RSSI_THRESHOLD_DEFAULT,
        show_default=True,
        callback=_validate_cli_finite_float,
        help="ランドマーク検出とみなす RSSI の下限 [dBm]",
    )(f)
    f = click.option(
        "--ble-data",
        "ble_data_path",
        default=_BLE_DATA_DEFAULT,
        type=click.Path(),
        show_default=True,
        help="BLE RSSI CSV のパス",
    )(f)
    f = click.option(
        "--ble-landmark/--no-ble-landmark",
        default=_BLE_LANDMARK_DEFAULT,
        show_default=True,
        help="BLE ランドマークによる位置補正の有効・無効",
    )(f)
    f = click.option(
        "--sidestep-smoothing",
        type=click.Choice(_SIDESTEP_SMOOTHING_CHOICES),
        default=_SIDESTEP_SMOOTHING_DEFAULT,
        show_default=True,
        help="横歩き判定の平滑化",
    )(f)
    f = click.option(
        "--forward-heading-source",
        type=click.Choice(_FORWARD_HEADING_SOURCE_CHOICES),
        default=_FORWARD_HEADING_SOURCE_DEFAULT,
        show_default=True,
        help="forward 判定ステップの軌跡方位ソース",
    )(f)
    f = click.option(
        "--sidestep-heading-source",
        type=click.Choice(_SIDESTEP_HEADING_SOURCE_CHOICES),
        default=_SIDESTEP_HEADING_SOURCE_DEFAULT,
        show_default=True,
        help="確定横歩きステップの軌跡方位ソース",
    )(f)
    f = click.option(
        "--sidestep-suspect-mode",
        type=click.Choice(_SIDESTEP_SUSPECT_MODE_CHOICES),
        default=_SIDESTEP_SUSPECT_MODE_DEFAULT,
        show_default=True,
        help="横歩き疑いステップの軌跡反映モード",
    )(f)
    f = click.option(
        "--motion-heading-correction",
        type=click.Choice(_MOTION_HEADING_CORRECTION_CHOICES),
        default=_MOTION_HEADING_CORRECTION_DEFAULT,
        show_default=True,
        help="水平加速度移動方向の固定ずれ補正",
    )(f)
    f = click.option(
        "--sidestep-min-lateral-displacement",
        type=float,
        callback=_validate_cli_non_negative_float,
        default=_SIDESTEP_MIN_LATERAL_DISPLACEMENT_DEFAULT,
        show_default=True,
        help="横歩き判定に必要な横方向変位の最小値 [m]",
    )(f)
    f = click.option(
        "--sidestep-lateral-ratio",
        type=float,
        callback=_validate_cli_positive_float,
        default=_SIDESTEP_LATERAL_RATIO_DEFAULT,
        show_default=True,
        help="横歩き判定に使う 横方向/前方向 の最小比率",
    )(f)
    f = click.option(
        "--step-length-method",
        type=click.Choice(_STEP_LENGTH_CHOICES),
        default=_STEP_LENGTH_DEFAULT,
        show_default=True,
        help="歩幅推定手法",
    )(f)
    f = click.option(
        "--step-detection",
        type=click.Choice(_STEP_DETECTION_CHOICES),
        default=_STEP_DETECTION_DEFAULT,
        show_default=True,
        help="ステップ検出手法",
    )(f)
    f = click.option(
        "--heading-method",
        type=click.Choice(_HEADING_METHOD_CHOICES),
        default=_HEADING_METHOD_DEFAULT,
        show_default=True,
        help="方位推定手法",
    )(f)
    f = click.option(
        "--gyro-bias-method",
        type=click.Choice(_GYRO_BIAS_METHOD_CHOICES),
        default=_GYRO_BIAS_METHOD_DEFAULT,
        show_default=True,
        help="ジャイロバイアス推定手法",
    )(f)
    f = click.option(
        "--gyro-bias",
        type=float,
        callback=_validate_cli_finite_float,
        default=None,
        help="manual 指定時のジャイロバイアス [rad/s]",
    )(f)
    f = click.option(
        "--direction",
        type=float,
        callback=_validate_cli_finite_float,
        default=_DIRECTION_DEFAULT,
        show_default=True,
        help="歩行開始方向のオフセット [度]",
    )(f)
    f = click.option(
        "--height-m",
        type=float,
        callback=_validate_cli_positive_float,
        default=_HEIGHT_DEFAULT,
        show_default=True,
        help="歩幅推定に使うユーザー身長 [m]",
    )(f)
    f = click.option(
        "--scale",
        type=float,
        callback=_validate_cli_scale,
        default=_SCALE_DEFAULT,
        show_default=True,
        help="1ピクセルあたりのメートル数",
    )(f)
    f = click.option(
        "--origin-px",
        nargs=2,
        type=int,
        default=_ORIGIN_DEFAULT,
        show_default=True,
        help="軌跡起点のピクセル座標 X Y",
    )(f)
    f = click.option(
        "--floormap",
        "-f",
        default=_FLOORMAP_DEFAULT,
        type=click.Path(),
        show_default=True,
        help="フロアマップ画像のパス",
    )(f)
    f = click.option(
        "--data-dir",
        "-d",
        default=_DATA_DIR_DEFAULT,
        type=click.Path(),
        show_default=True,
        help="入力データフォルダ",
    )(f)
    return f


@click.group()
def cli() -> None:
    """rikka — PDR 歩行軌跡推定ツール"""


# ``rikka.cli`` サブパッケージの import 後も Click グループを参照できるよう保持する。
_click_cli = cli


def _run_pdr(
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
    ble_data_path: str,
    ble_rssi_threshold: float,
) -> None:
    from ..common.lib.sensors import load_sensor_data  # noqa: PLC0415
    from .commands import run as _run  # noqa: PLC0415

    _validate_gyro_bias_options(gyro_bias_method, gyro_bias)
    df_acc, df_gyro = load_sensor_data(data_dir)
    _run(
        df_acc=df_acc,
        df_gyro=df_gyro,
        plot=not no_plot,
        use_particle_filter=False,
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
        ble_data_path=ble_data_path,
        ble_rssi_threshold=ble_rssi_threshold,
    )


@cli.command()
@_common_options
def run(
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
    ble_data_path: str,
    ble_rssi_threshold: float,
) -> None:
    """決定論的 PDR で歩行軌跡を推定する。"""
    _run_pdr(
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
        ble_landmark,
        ble_data_path,
        ble_rssi_threshold,
    )


cli.add_command(run, name="pdr")


@cli.command()
@click.option(
    "--pf-landmark-mode",
    type=click.Choice(_PF_LANDMARK_MODE_CHOICES),
    default=_PF_LANDMARK_MODE_DEFAULT,
    show_default=True,
    help="particle filter へのランドマーク反映方式",
)
@click.option(
    "--pf-path-selection",
    type=click.Choice(_PF_PATH_SELECTION_CHOICES),
    default=_PF_PATH_SELECTION_DEFAULT,
    show_default=True,
    help="PFの代表軌跡選択方式",
)
@click.option(
    "--motion-predictive-weight-power",
    type=float,
    callback=_validate_cli_non_negative_float,
    default=_PF_MOTION_PREDICTIVE_WEIGHT_POWER_DEFAULT,
    show_default=True,
    help="運動状態の予測尤度をPF重みに掛ける指数（0で無効）",
)
@click.option(
    "--pf-particles",
    type=click.IntRange(min=1),
    default=_PF_NUM_PARTICLES_DEFAULT,
    show_default=True,
    help="パーティクルフィルタで使用する粒子数",
)
@click.option(
    "--pf-seed",
    type=int,
    default=None,
    help="パーティクルフィルタ乱数の seed（回帰検証用）",
)
@click.option(
    "--save-path-comparison",
    is_flag=True,
    default=False,
    help="代表軌跡候補の比較図を保存",
)
@click.option(
    "--step-frames-dpi",
    type=click.IntRange(min=1),
    default=_PF_STEP_FRAMES_DPI_DEFAULT,
    show_default=True,
    help="段階別画像と代表軌跡比較図の解像度",
)
@click.option(
    "--step-frames-arrows",
    type=click.IntRange(min=0),
    default=_PF_STEP_FRAMES_ARROWS_DEFAULT,
    show_default=True,
    help="段階別画像へ描く重み上位の方位矢印数",
)
@click.option(
    "--step-frames-range",
    type=int,
    nargs=2,
    default=None,
    callback=_validate_cli_step_frames_range,
    metavar="A B",
    help="保存する歩の範囲（1始まり、両端含む）",
)
@click.option(
    "--save-step-frames",
    is_flag=True,
    default=False,
    help="1歩ごとの段階別パーティクル画像を保存",
)
@click.option(
    "--save-animation",
    is_flag=True,
    default=False,
    help="--no-plot 指定時もパーティクルフィルタのアニメーションを保存",
)
@_common_options
def particle(
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
    ble_data_path: str,
    ble_rssi_threshold: float,
    save_animation: bool,
    save_step_frames: bool,
    step_frames_range: tuple[int, int] | None,
    step_frames_arrows: int,
    step_frames_dpi: int,
    save_path_comparison: bool,
    pf_particles: int,
    pf_seed: int | None,
    motion_predictive_weight_power: float,
    pf_path_selection: str,
    pf_landmark_mode: str,
) -> None:
    """パーティクルフィルタ + マップマッチングで歩行軌跡を推定する。"""
    from ..common.lib.sensors import load_sensor_data  # noqa: PLC0415
    from .commands import run as _run  # noqa: PLC0415

    _validate_gyro_bias_options(gyro_bias_method, gyro_bias)
    df_acc, df_gyro = load_sensor_data(data_dir)
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
        motion_predictive_weight_power=motion_predictive_weight_power,
        pf_path_selection=pf_path_selection,
        pf_landmark_mode=pf_landmark_mode,
        ble_landmark=ble_landmark,
        ble_data_path=ble_data_path,
        ble_rssi_threshold=ble_rssi_threshold,
    )


@cli.command()
@click.option(
    "--data-dir",
    "-d",
    default=_DATA_DIR_DEFAULT,
    type=click.Path(),
    show_default=True,
    help="入力データフォルダ",
)
@click.option(
    "--step-detection",
    type=click.Choice(_STEP_DETECTION_CHOICES),
    default=_STEP_DETECTION_DEFAULT,
    show_default=True,
    help="ステップ検出手法",
)
@click.option(
    "--gyro-bias-method",
    type=click.Choice(_GYRO_BIAS_METHOD_CHOICES),
    default=_GYRO_BIAS_METHOD_DEFAULT,
    show_default=True,
    help="ジャイロバイアス推定手法",
)
@click.option(
    "--gyro-bias",
    type=float,
    callback=_validate_cli_finite_float,
    default=None,
    help="manual 指定時のジャイロバイアス [rad/s]",
)
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
@click.option(
    "--mode",
    type=click.Choice(BLE_SAMPLE_MODES),
    default=_BLE_SAMPLE_MODE_DEFAULT,
    show_default=True,
    help="RSSI生成方式",
)
@click.option(
    "--source",
    type=click.Choice(BLE_SAMPLE_SOURCES),
    default="pdr",
    show_default=True,
    help="distance方式で使う歩行者位置の取得元",
)
@click.option(
    "--truth-csv",
    default=_BLE_SAMPLE_TRUTH_DEFAULT,
    type=click.Path(),
    show_default=True,
    help="source=truth で使う正解軌跡CSV",
)
@click.option(
    "--data-dir",
    "-d",
    default=_DATA_DIR_DEFAULT,
    type=click.Path(),
    show_default=True,
    help="時間軸の基準にする入力データフォルダ",
)
@click.option(
    "--output",
    "-o",
    default=_BLE_DATA_DEFAULT,
    type=click.Path(),
    show_default=True,
    help="生成する BLE RSSI CSV の保存先",
)
@click.option(
    "--seed",
    type=int,
    default=_BLE_SAMPLE_SEED_DEFAULT,
    show_default=True,
    help="サンプル生成の乱数シード",
)
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
    from ..common.config import (  # noqa: PLC0415
        FLOORMAP_ORIGIN_PX,
        FLOORMAP_SCALE,
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


def main() -> None:
    cli()
