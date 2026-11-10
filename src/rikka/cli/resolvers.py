"""CLI の計測条件とランドマーク入力を解決する。

役割:
    計測情報と明示引数から入力条件とランドマーク経路を決める。
依存元:
    common の設定・共有型と BLE の読み込み処理を使用する。
利用先:
    options の各コマンドが解析入力の準備に使用する。
処理フロー:
    明示値を優先して入力を補い、経路の座標と順序を返す。
"""

from pathlib import Path

import click

from ..common.config import (
    BLE_DATA_PATH,
    BLE_LANDMARK_ANCHORS,
)
from ..common.lib.models import Landmark
from ..common.settings import merge_landmark_anchors

_BLE_DATA_DEFAULT = BLE_DATA_PATH


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
    if ble_data_path == _BLE_DATA_DEFAULT and measurement_dir.is_dir():
        local_names = {
            path.name for path in measurement_dir.iterdir() if path.is_file()
        }
        for name in ("BLE.csv", "ble.csv"):
            local_data_path = measurement_dir / name
            if name in local_names:
                resolved_data_path = local_data_path
                break

    local_position_path = resolved_data_path.with_name("BLE_pos.csv")
    if not local_position_path.is_file():
        from ..ble.lib.loader import is_logger_ble_data  # noqa: PLC0415

        if resolved_data_path.is_file() and is_logger_ble_data(resolved_data_path):
            raise ValueError(
                "Thingsup形式のBLEログには同じディレクトリの BLE_pos.csv が必要です: "
                f"{resolved_data_path}"
            )
        local_metadata_path = resolved_data_path.with_name("metadata.json")
        if local_metadata_path.is_file():
            from ..ble.lib.loader import load_ble_metadata_landmarks  # noqa: PLC0415

            metadata_landmarks = load_ble_metadata_landmarks(local_metadata_path)
            if metadata_landmarks is not None:
                if not metadata_landmarks:
                    raise ValueError(
                        "metadata.jsonに有効なBLE配置がありません: "
                        f"{local_metadata_path}"
                    )
                return str(resolved_data_path), merge_landmark_anchors(
                    metadata_landmarks,
                    BLE_LANDMARK_ANCHORS,
                    coordinate_source="metadata.json",
                )
        return str(resolved_data_path), None

    from ..ble.lib.loader import load_ble_landmarks  # noqa: PLC0415

    landmarks = merge_landmark_anchors(
        load_ble_landmarks(local_position_path),
        BLE_LANDMARK_ANCHORS,
        coordinate_source="BLE_pos.csv",
    )
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
    """CLI明示値を優先し、計測CSVまたはmetadataの開始条件を反映する。"""
    import matplotlib.image as mpimg  # noqa: PLC0415
    from click.core import ParameterSource  # noqa: PLC0415

    from ..common.lib.measurement_config import (  # noqa: PLC0415
        load_measurement_config,
    )

    config_path = Path(data_dir) / "walk_config.csv"
    android_metadata = all(
        (Path(data_dir) / name).is_file()
        for name in ("acce.csv", "gyro.csv", "metadata.json")
    )
    if not config_path.is_file() and not android_metadata:
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


def _resolve_free_origin_settings(
    ctx: click.Context,
    data_dir: str,
    direction: float,
    height_m: float,
    ble_landmark: bool,
    option_label: str = "--free-origin",
    *,
    allow_ble_landmark: bool = False,
) -> tuple[tuple[int, int], float, float]:
    """起点・方位を固定し、身長だけ反映する。BLE併用は明示的な許可時のみ認める。"""
    from click.core import ParameterSource  # noqa: PLC0415

    from ..common.lib.measurement_config import (  # noqa: PLC0415
        load_measurement_user_height,
    )

    for name, label in (("origin_px", "--origin-px"), ("direction", "--direction")):
        if ctx.get_parameter_source(name) is ParameterSource.COMMANDLINE:
            raise click.UsageError(
                f"{option_label} と {label} は同時に指定できません。"
            )
    if ble_landmark and not allow_ble_landmark:
        reason = (
            "地図座標を使わない"
            if option_label == "--free-origin"
            else "初期位置を使わない"
        )
        raise click.UsageError(
            f"{option_label} は{reason}ため --ble-landmark と併用できません。"
        )

    # 起点・初期方位を使わないため、計測設定からは身長だけを読む。
    # 起点や初期方位を持たない計測でも実行できるようにするための分離。
    if ctx.get_parameter_source("height_m") is not ParameterSource.COMMANDLINE:
        measured_height = load_measurement_user_height(data_dir)
        if measured_height is not None:
            height_m = measured_height
    return (0, 0), 0.0, height_m


def _parse_landmark_specs(
    specs: tuple[str, ...],
) -> tuple[tuple[str, float, float], ...]:
    """``beacon_id:X:Y`` 形式の指定を経路順のランドマーク座標へ変換する。"""
    landmarks: list[tuple[str, float, float]] = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3 or not parts[0].strip():
            raise click.UsageError(
                "--landmark は beacon_id:ピクセルX:ピクセルY の形式で指定してください"
                f": {spec}"
            )
        try:
            pixel_x = float(parts[1])
            pixel_y = float(parts[2])
        except ValueError as exc:
            raise click.UsageError(
                f"--landmark のピクセル座標は数値で指定してください: {spec}"
            ) from exc
        landmarks.append((parts[0].strip(), pixel_x, pixel_y))
    return tuple(landmarks)


def _resolve_landmark_path_inputs(
    data_dir: str,
    landmark_specs: tuple[str, ...],
) -> tuple[tuple[tuple[str, float, float], ...], str, str | None]:
    """ランドマーク座標の出所を、明示指定・計測入力・設定既定の順に解決する。"""
    if landmark_specs:
        return _parse_landmark_specs(landmark_specs), "--landmark の指定", None

    from ..common.config import BLE_LANDMARKS_PX  # noqa: PLC0415

    ble_data_path, landmarks = _resolve_ble_inputs(data_dir, True, _BLE_DATA_DEFAULT)
    if landmarks:
        source = (
            "BLE_pos.csv"
            if Path(ble_data_path).with_name("BLE_pos.csv").is_file()
            else "metadata.json"
        )
        return (
            tuple((item.beacon_id, item.pixel_x, item.pixel_y) for item in landmarks),
            f"{source}（{data_dir}）",
            ble_data_path,
        )
    return BLE_LANDMARKS_PX, "設定既定の BLE_LANDMARKS_PX（合成サンプル用）", None


def _resolve_landmark_path_order(
    ble_data_path: str | None,
    landmarks: tuple[tuple[str, float, float], ...],
) -> dict[str, float] | None:
    """計測のBLEログがある場合に、平滑RSSIのピーク時刻を経路順として返す。

    BLEログが無い場合だけ ``None`` を返す。ログはあるが既知ランドマークの受信が
    1件も無い場合は空の辞書を返し、指定順への無言のフォールバックを避ける。
    """
    if ble_data_path is None or not Path(ble_data_path).is_file():
        return None

    from ..ble.pipeline import run_ble_peak_times  # noqa: PLC0415
    from ..common.settings import BleLandmarkSettings  # noqa: PLC0415

    settings = BleLandmarkSettings(
        enabled=True,
        data_path=ble_data_path,
        landmarks=tuple(
            Landmark(beacon_id=beacon_id, pixel_x=pixel_x, pixel_y=pixel_y)
            for beacon_id, pixel_x, pixel_y in landmarks
        ),
    )
    return run_ble_peak_times(settings)


def _report_landmarks_without_passage(
    landmarks: tuple[tuple[str, float, float], ...],
    order_times: dict[str, float],
    source: str,
) -> None:
    """BLE受信が無く経路から外れるランドマークを知らせ、2点未満なら中止する。"""
    dropped = tuple(
        beacon_id for beacon_id, _, _ in landmarks if beacon_id not in order_times
    )
    if len(landmarks) - len(dropped) < 2:
        raise click.UsageError(
            "BLEログに受信があるランドマークが2点未満のため経路になりません"
            f"（出所: {source}、受信なし: {', '.join(dropped)}）。"
            "指定順で結ぶ場合は --order given を使ってください。"
        )
    if dropped:
        click.echo(f"BLE受信が無いため経路から除外: {', '.join(dropped)}")
