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
from pathlib import Path

import click

from ..common.config import (
    BLE_DATA_PATH,
    BLE_LANDMARK_ENABLED,
    BLE_MAX_CORRECTION_M,
    BLE_MAX_WARP_SPAN_M,
    BLE_PDR_CORRECTION_MODE,
    BLE_PREFLIGHT_MODE,
    BLE_RETROFIT_DAMP_FACTORS,
    BLE_RETROFIT_FORWARD_MODE,
    BLE_RETROFIT_MAP_CHECK,
    BLE_RETROFIT_MAX_HEADING_DEG,
    BLE_RETROFIT_MIN_SPAN_M,
    BLE_RETROFIT_STRIDE_SCALE_MAX,
    BLE_RETROFIT_STRIDE_SCALE_MIN,
    BLE_RSSI_RELEASE_MARGIN_DB,
    BLE_RSSI_RELEASE_STREAK,
    BLE_RSSI_THRESHOLD_DBM,
    BLE_SAMPLE_MODE,
    BLE_SAMPLE_SEED,
    BLE_SAMPLE_TRUTH_PATH,
    BLE_SYNC_WINDOW_S,
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
    PF_LANDMARK_RETROFIT,
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
from ..common.lib.models import Landmark
from ..common.lib.validation import (
    BLE_PDR_CORRECTION_MODES,
    BLE_PREFLIGHT_MODES,
    BLE_RETROFIT_FORWARD_MODES,
    BLE_RETROFIT_MAP_CHECK_MODES,
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
_BLE_CORRECTION_DEFAULT = BLE_PDR_CORRECTION_MODE
_BLE_CORRECTION_CHOICES = BLE_PDR_CORRECTION_MODES
_BLE_PREFLIGHT_DEFAULT = BLE_PREFLIGHT_MODE
_BLE_PREFLIGHT_CHOICES = BLE_PREFLIGHT_MODES
_BLE_MAX_CORRECTION_DEFAULT = BLE_MAX_CORRECTION_M
_BLE_MAX_WARP_SPAN_DEFAULT = BLE_MAX_WARP_SPAN_M
_BLE_RETROFIT_FORWARD_DEFAULT = BLE_RETROFIT_FORWARD_MODE
_BLE_RETROFIT_MAX_HEADING_DEFAULT = BLE_RETROFIT_MAX_HEADING_DEG
_BLE_RETROFIT_SCALE_MIN_DEFAULT = BLE_RETROFIT_STRIDE_SCALE_MIN
_BLE_RETROFIT_SCALE_MAX_DEFAULT = BLE_RETROFIT_STRIDE_SCALE_MAX
_BLE_RETROFIT_MIN_SPAN_DEFAULT = BLE_RETROFIT_MIN_SPAN_M
_BLE_RETROFIT_MAP_CHECK_DEFAULT = BLE_RETROFIT_MAP_CHECK
_BLE_RETROFIT_DAMP_FACTORS_DEFAULT = BLE_RETROFIT_DAMP_FACTORS
_BLE_RELEASE_MARGIN_DEFAULT = BLE_RSSI_RELEASE_MARGIN_DB
_BLE_RELEASE_STREAK_DEFAULT = BLE_RSSI_RELEASE_STREAK
_BLE_SYNC_WINDOW_DEFAULT = BLE_SYNC_WINDOW_S
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
_PF_LANDMARK_RETROFIT_DEFAULT = PF_LANDMARK_RETROFIT
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


def _resolve_ble_inputs(
    data_dir: str,
    ble_landmark: bool,
    ble_data_path: str,
) -> tuple[str, tuple[Landmark, ...] | None]:
    """計測ディレクトリに同居するBLEログと既知座標を優先して解決する。"""
    if not ble_landmark:
        return ble_data_path, None

    measurement_dir = Path(data_dir)
    resolved_data_path = Path(ble_data_path)
    local_data_path = measurement_dir / "BLE.csv"
    if ble_data_path == _BLE_DATA_DEFAULT and local_data_path.is_file():
        resolved_data_path = local_data_path

    local_position_path = resolved_data_path.with_name("BLE_pos.csv")
    if not local_position_path.is_file():
        from ..ble.lib.loader import is_logger_ble_data  # noqa: PLC0415

        if resolved_data_path.is_file() and is_logger_ble_data(resolved_data_path):
            raise ValueError(
                "Thingsup形式のBLEログには同じディレクトリの BLE_pos.csv が必要です: "
                f"{resolved_data_path}"
            )
        return str(resolved_data_path), None

    from ..ble.lib.loader import load_ble_landmarks  # noqa: PLC0415

    landmarks = load_ble_landmarks(local_position_path)
    if not landmarks:
        raise ValueError(
            f"BLE_pos.csv に座標が確定した端末がありません: {local_position_path}"
        )
    return str(resolved_data_path), landmarks


def _resolve_measurement_settings(
    ctx: click.Context,
    data_dir: str,
    floormap: str,
    origin_px: tuple[int, int],
    direction: float,
    height_m: float,
) -> tuple[tuple[int, int], float, float]:
    """CLI明示値を優先しつつ walk_config.csv の計測条件を反映する。"""
    import matplotlib.image as mpimg  # noqa: PLC0415
    from click.core import ParameterSource  # noqa: PLC0415

    from ..common.lib.measurement_config import (  # noqa: PLC0415
        load_measurement_config,
    )

    config_path = Path(data_dir) / "walk_config.csv"
    if not config_path.is_file():
        return origin_px, direction, height_m
    try:
        image = mpimg.imread(floormap)
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError(
            f"walk_config.csv の検証用フロアマップを読めません: {floormap}"
        ) from exc
    config = load_measurement_config(
        data_dir,
        image_size_px=(int(image.shape[1]), int(image.shape[0])),
    )
    if config is None:  # pragma: no cover - 直前の存在確認との防御境界
        return origin_px, direction, height_m
    if ctx.get_parameter_source("origin_px") is not ParameterSource.COMMANDLINE:
        origin_px = config.origin_px
    if ctx.get_parameter_source("direction") is not ParameterSource.COMMANDLINE:
        direction = config.initial_direction_deg
    if (
        config.user_height_m is not None
        and ctx.get_parameter_source("height_m") is not ParameterSource.COMMANDLINE
    ):
        height_m = config.user_height_m
    return origin_px, direction, height_m


def _common_options(f: click.decorators.FC) -> click.decorators.FC:
    """run / particle コマンド共通オプションをまとめたデコレータ。"""
    f = click.option(
        "--ble-retrofit-damp-factor",
        "ble_retrofit_damp_factors",
        type=float,
        multiple=True,
        default=_BLE_RETROFIT_DAMP_FACTORS_DEFAULT,
        show_default=True,
        callback=lambda ctx, param, value: tuple(
            _validate_cli_positive_float(ctx, param, item) for item in value
        ),
        help="地図違反時に試す相似補正の減衰係数（複数指定可）",
    )(f)
    f = click.option(
        "--ble-retrofit-map-check",
        type=click.Choice(BLE_RETROFIT_MAP_CHECK_MODES),
        default=_BLE_RETROFIT_MAP_CHECK_DEFAULT,
        show_default=True,
        help="相似補正後の壁交差の扱い",
    )(f)
    f = click.option(
        "--ble-retrofit-min-span",
        type=float,
        default=_BLE_RETROFIT_MIN_SPAN_DEFAULT,
        show_default=True,
        callback=_validate_cli_positive_float,
        help="相似補正を解く最小アンカー間距離 [m]",
    )(f)
    f = click.option(
        "--ble-retrofit-stride-scale-max",
        type=float,
        default=_BLE_RETROFIT_SCALE_MAX_DEFAULT,
        show_default=True,
        callback=_validate_cli_positive_float,
        help="相似補正で許容する歩幅倍率の上限",
    )(f)
    f = click.option(
        "--ble-retrofit-stride-scale-min",
        type=float,
        default=_BLE_RETROFIT_SCALE_MIN_DEFAULT,
        show_default=True,
        callback=_validate_cli_positive_float,
        help="相似補正で許容する歩幅倍率の下限",
    )(f)
    f = click.option(
        "--ble-retrofit-max-heading",
        type=float,
        default=_BLE_RETROFIT_MAX_HEADING_DEFAULT,
        show_default=True,
        callback=_validate_cli_non_negative_float,
        help="相似補正で許容する絶対回転角の上限 [deg]",
    )(f)
    f = click.option(
        "--ble-retrofit-forward",
        type=click.Choice(BLE_RETROFIT_FORWARD_MODES),
        default=_BLE_RETROFIT_FORWARD_DEFAULT,
        show_default=True,
        help="相似補正を後続歩へ保持するか固定するか",
    )(f)
    f = click.option(
        "--ble-max-warp-span",
        type=float,
        default=_BLE_MAX_WARP_SPAN_DEFAULT,
        show_default=True,
        callback=_validate_cli_positive_float,
        help="warpで一度に配分する軌跡区間の上限 [m]",
    )(f)
    f = click.option(
        "--ble-max-correction",
        type=float,
        default=_BLE_MAX_CORRECTION_DEFAULT,
        show_default=True,
        callback=_validate_cli_positive_float,
        help="1回のBLE補正移動量の上限 [m]",
    )(f)
    f = click.option(
        "--ble-preflight",
        type=click.Choice(_BLE_PREFLIGHT_CHOICES),
        default=_BLE_PREFLIGHT_DEFAULT,
        show_default=True,
        help="RSSI距離整合FAIL時の扱い",
    )(f)
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
        "--ble-sync-window",
        type=float,
        default=_BLE_SYNC_WINDOW_DEFAULT,
        show_default=True,
        callback=_validate_cli_non_negative_float,
        help="同時受信としてまとめる時刻窓 [s]",
    )(f)
    f = click.option(
        "--ble-release-streak",
        type=click.IntRange(min=1),
        default=_BLE_RELEASE_STREAK_DEFAULT,
        show_default=True,
        help="接近ラッチ解除に必要な連続観測数",
    )(f)
    f = click.option(
        "--ble-release-margin",
        type=float,
        default=_BLE_RELEASE_MARGIN_DEFAULT,
        show_default=True,
        callback=_validate_cli_non_negative_float,
        help="接近ラッチ解除のRSSI余裕 [dB]",
    )(f)
    f = click.option(
        "--ble-correction",
        type=click.Choice(_BLE_CORRECTION_CHOICES),
        default=_BLE_CORRECTION_DEFAULT,
        show_default=True,
        help="通常PDRへのランドマーク補正方式",
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
    ble_data_path: str,
    ble_rssi_threshold: float,
    ble_correction: str,
    ble_release_margin: float,
    ble_release_streak: int,
    ble_sync_window: float,
    ble_preflight: str,
    ble_max_correction: float,
    ble_max_warp_span: float,
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
    ble_landmark: bool,
    ble_data_path: str,
    ble_rssi_threshold: float,
    ble_correction: str,
    ble_release_margin: float,
    ble_release_streak: int,
    ble_sync_window: float,
    ble_preflight: str,
    ble_max_correction: float,
    ble_max_warp_span: float,
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
        ble_landmark,
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
@click.option(
    "--pf-landmark-retrofit/--no-pf-landmark-retrofit",
    default=_PF_LANDMARK_RETROFIT_DEFAULT,
    show_default=True,
    help="PF代表軌跡のランドマーク不連続を相似補正",
)
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
    ble_data_path: str,
    ble_rssi_threshold: float,
    ble_correction: str,
    ble_release_margin: float,
    ble_release_streak: int,
    ble_sync_window: float,
    ble_preflight: str,
    ble_max_correction: float,
    ble_max_warp_span: float,
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
    pf_particles: int,
    pf_seed: int | None,
    motion_predictive_weight_power: float,
    pf_path_selection: str,
    pf_landmark_mode: str,
    pf_landmark_retrofit: bool,
) -> None:
    """パーティクルフィルタ + マップマッチングで歩行軌跡を推定する。"""
    from ..common.lib.sensors import load_sensor_data  # noqa: PLC0415
    from .commands import run as _run  # noqa: PLC0415

    _validate_gyro_bias_options(gyro_bias_method, gyro_bias)
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
        pf_landmark_retrofit=pf_landmark_retrofit,
        ble_landmark=ble_landmark,
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
