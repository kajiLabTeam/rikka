import click

from .config import (
    DATA_DIR,
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    GYRO_BIAS_METHOD,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    STEP_DETECTION_METHOD,
    USER_HEIGHT_M,
)
from .ping import ping as ping

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


def _validate_cli_scale(
    _ctx: click.Context,
    _param: click.Parameter,
    value: float,
) -> float:
    """scale が正の値であることを確認する。"""
    if value <= 0:
        raise click.BadParameter("scale は正の値を指定してください。")
    return value


def _common_options(f: click.decorators.FC) -> click.decorators.FC:
    """run / particle コマンド共通オプションをまとめたデコレータ。"""
    f = click.option(
        "--no-plot", is_flag=True, default=False, help="グラフ表示を無効化"
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
        default=None,
        help="manual 指定時のジャイロバイアス [rad/s]",
    )(f)
    f = click.option(
        "--direction",
        type=float,
        default=_DIRECTION_DEFAULT,
        show_default=True,
        help="歩行開始方向のオフセット [度]",
    )(f)
    f = click.option(
        "--height-m",
        type=float,
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
    no_plot: bool,
) -> None:
    from .analyze.pdr import load_sensor_data  # noqa: PLC0415
    from .analyze.pdr import run as _run  # noqa: PLC0415

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
        no_plot,
    )


@cli.command()
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
    no_plot: bool,
    save_animation: bool,
) -> None:
    """パーティクルフィルタ + マップマッチングで歩行軌跡を推定する。"""
    from .analyze.pdr import load_sensor_data  # noqa: PLC0415
    from .analyze.pdr import run as _run  # noqa: PLC0415

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

    plot_sensor_data(
        data_dir,
        step_detection_method=step_detection,
        gyro_bias_method=gyro_bias_method,
        gyro_bias=gyro_bias,
    )


def main() -> None:
    cli()
