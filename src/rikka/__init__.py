import click

from .config import (
    DATA_DIR,
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    INITIAL_DIRECTION,
)

_DATA_DIR_DEFAULT = DATA_DIR
_FLOORMAP_DEFAULT = FLOORMAP_PATH
_ORIGIN_DEFAULT = FLOORMAP_ORIGIN_PX
_SCALE_DEFAULT = FLOORMAP_SCALE
_DIRECTION_DEFAULT = INITIAL_DIRECTION


def _common_options(f: click.decorators.FC) -> click.decorators.FC:
    """run / particle コマンド共通オプションをまとめたデコレータ。"""
    f = click.option(
        "--no-plot", is_flag=True, default=False, help="グラフ表示を無効化"
    )(f)
    f = click.option(
        "--direction",
        type=float,
        default=_DIRECTION_DEFAULT,
        show_default=True,
        help="歩行開始方向のオフセット [度]",
    )(f)
    f = click.option(
        "--scale",
        type=float,
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


@click.group()  # type: ignore[misc]
def cli() -> None:
    """rikka — PDR 歩行軌跡推定ツール"""


def _run_pdr(
    data_dir: str,
    floormap: str,
    origin_px: tuple[int, int],
    scale: float,
    direction: float,
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
    )


@cli.command()  # type: ignore[misc]
@_common_options
def run(
    data_dir: str,
    floormap: str,
    origin_px: tuple[int, int],
    scale: float,
    direction: float,
    no_plot: bool,
) -> None:
    """決定論的 PDR で歩行軌跡を推定する。"""
    _run_pdr(data_dir, floormap, origin_px, scale, direction, no_plot)


@cli.command()  # type: ignore[misc]
@_common_options
def pdr(
    data_dir: str,
    floormap: str,
    origin_px: tuple[int, int],
    scale: float,
    direction: float,
    no_plot: bool,
) -> None:
    """決定論的 PDR で歩行軌跡を推定する（run の別名）。"""
    _run_pdr(data_dir, floormap, origin_px, scale, direction, no_plot)


@cli.command()  # type: ignore[misc]
@_common_options
def particle(
    data_dir: str,
    floormap: str,
    origin_px: tuple[int, int],
    scale: float,
    direction: float,
    no_plot: bool,
) -> None:
    """パーティクルフィルタ + マップマッチングで歩行軌跡を推定する。"""
    from .analyze.pdr import load_sensor_data  # noqa: PLC0415
    from .analyze.pdr import run as _run  # noqa: PLC0415

    df_acc, df_gyro = load_sensor_data(data_dir)
    _run(
        df_acc=df_acc,
        df_gyro=df_gyro,
        plot=not no_plot,
        use_particle_filter=True,
        floormap_path=floormap,
        origin_px=origin_px,
        scale=scale,
        initial_direction=direction,
    )


@cli.command()  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "--data-dir",
    "-d",
    default=_DATA_DIR_DEFAULT,
    type=click.Path(),
    show_default=True,
    help="入力データフォルダ",
)
def sensor(data_dir: str) -> None:
    """センサーデータをグラフ化して入力フォルダに保存する。"""
    from .analyze.sensor_plot import plot_sensor_data  # noqa: PLC0415

    plot_sensor_data(data_dir)


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="バインドするホスト")
@click.option("--port", default=8000, show_default=True, help="待ち受けポート番号")
def serve(host: str, port: int) -> None:
    """接続確認用 HTTP サーバーを起動する。"""
    import uvicorn  # noqa: PLC0415

    from .server import app  # noqa: PLC0415

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    cli()
