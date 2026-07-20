"""rikka の CLI エントリポイント。

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

from .config import (
    DATA_DIR,
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    GYRO_BIAS_METHOD,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    MOTION_ESTIMATION,
    PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    PF_PATH_SELECTION,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    SIDESTEP_SUSPECT_MODE,
    SMOOTHING_MODE,
    STEP_DETECTION_METHOD,
    USER_HEIGHT_M,
)
from .matplotlib_config import configure_matplotlib_cache
from .ping import ping as ping

configure_matplotlib_cache()

_DATA_DIR_DEFAULT = DATA_DIR
_FLOORMAP_DEFAULT = FLOORMAP_PATH
_ORIGIN_DEFAULT = FLOORMAP_ORIGIN_PX
_SCALE_DEFAULT = FLOORMAP_SCALE
_DIRECTION_DEFAULT = INITIAL_DIRECTION
_HEIGHT_DEFAULT = USER_HEIGHT_M
_STEP_DETECTION_DEFAULT = STEP_DETECTION_METHOD
_STEP_DETECTION_CHOICES = ("peak", "paper_vertical_threshold")
_HEADING_METHOD_DEFAULT = HEADING_METHOD
_HEADING_METHOD_CHOICES = (
    "gyro",
    "accel_method1",
    "accel_method2",
    "gyro_accel_motion",
)
_GYRO_BIAS_METHOD_DEFAULT = GYRO_BIAS_METHOD
_GYRO_BIAS_METHOD_CHOICES = ("prewalk_robust", "initial_robust", "quietest", "manual")
_SIDESTEP_LATERAL_RATIO_DEFAULT = SIDESTEP_LATERAL_RATIO
_SIDESTEP_MIN_LATERAL_DISPLACEMENT_DEFAULT = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M
_MOTION_HEADING_CORRECTION_DEFAULT = "auto"
_MOTION_HEADING_CORRECTION_CHOICES = ("auto", "none")
_SIDESTEP_SMOOTHING_DEFAULT = SIDESTEP_SMOOTHING_METHOD
_SIDESTEP_SMOOTHING_CHOICES = ("none", "isolated", "clustered")
_FORWARD_HEADING_SOURCE_DEFAULT = FORWARD_HEADING_SOURCE
_FORWARD_HEADING_SOURCE_CHOICES = ("body", "motion")
_SIDESTEP_HEADING_SOURCE_DEFAULT = "motion"
_SIDESTEP_HEADING_SOURCE_CHOICES = ("motion", "body_lateral", "blend")
_SIDESTEP_SUSPECT_MODE_DEFAULT = SIDESTEP_SUSPECT_MODE
_SIDESTEP_SUSPECT_MODE_CHOICES = ("motion", "body_lateral", "blend", "forward")
_MOTION_ESTIMATION_DEFAULT = MOTION_ESTIMATION
_MOTION_ESTIMATION_CHOICES = ("legacy", "adaptive", "robust")
_SMOOTHING_MODE_DEFAULT = SMOOTHING_MODE
_SMOOTHING_MODE_CHOICES = ("causal", "offline")
_PF_MOTION_PREDICTIVE_WEIGHT_POWER_DEFAULT = PF_MOTION_PREDICTIVE_WEIGHT_POWER
_PF_PATH_SELECTION_DEFAULT = PF_PATH_SELECTION
_PF_PATH_SELECTION_CHOICES = ("current", "sequence")


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


def _run_pdr(
    data_dir: str,
    floormap: str,
    origin_px: tuple[int, int],
    scale: float,
    direction: float,
    height_m: float,
    step_detection: str,
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
) -> None:
    from .analyze.pdr import load_sensor_data  # noqa: PLC0415
    from .analyze.pdr import run as _run  # noqa: PLC0415

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
    )


@cli.command()
@_common_options
def pdr(
    data_dir: str,
    floormap: str,
    origin_px: tuple[int, int],
    scale: float,
    direction: float,
    height_m: float,
    step_detection: str,
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
) -> None:
    """決定論的 PDR で歩行軌跡を推定する（run の別名）。"""
    _run_pdr(
        data_dir,
        floormap,
        origin_px,
        scale,
        direction,
        height_m,
        step_detection,
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
    )


@cli.command()
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
    "--pf-seed",
    type=int,
    default=None,
    help="パーティクルフィルタ乱数の seed（回帰検証用）",
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
    save_animation: bool,
    pf_seed: int | None,
    motion_predictive_weight_power: float,
    pf_path_selection: str,
) -> None:
    """パーティクルフィルタ + マップマッチングで歩行軌跡を推定する。"""
    from .analyze.pdr import load_sensor_data  # noqa: PLC0415
    from .analyze.pdr import run as _run  # noqa: PLC0415

    _validate_gyro_bias_options(gyro_bias_method, gyro_bias)
    df_acc, df_gyro = load_sensor_data(data_dir)
    _run(
        df_acc=df_acc,
        df_gyro=df_gyro,
        plot=not no_plot,
        save_animation=True if save_animation else None,
        use_particle_filter=True,
        floormap_path=floormap,
        origin_px=origin_px,
        scale=scale,
        initial_direction=direction,
        height_m=height_m,
        step_detection_method=step_detection,
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
        motion_predictive_weight_power=motion_predictive_weight_power,
        pf_path_selection=pf_path_selection,
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
    from .analyze.sensor_plot import plot_sensor_data  # noqa: PLC0415

    _validate_gyro_bias_options(gyro_bias_method, gyro_bias)
    plot_sensor_data(
        data_dir,
        step_detection_method=step_detection,
        gyro_bias_method=gyro_bias_method,
        gyro_bias=gyro_bias,
    )


def main() -> None:
    cli()
