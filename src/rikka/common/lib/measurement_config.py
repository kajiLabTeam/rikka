"""計測ディレクトリ固有の歩行開始条件を読み込む。

役割:
    ``walk_config.csv`` の起点、初期方位、任意の身長を読み、値域を検証する。
    Androidロガー形式ではCSV未指定時にmetadata.jsonのstart制約を利用する。
依存元:
    ``common.lib.models.MeasurementConfig`` を返却型として利用する。
利用先:
    CLI が共通既定値より優先する計測条件を解決するために使用する。
処理フロー:
    CSV の存在確認、1行制約、数値変換、地図寸法を含む値域検証の順に処理する。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import MeasurementConfig

MEASUREMENT_CONFIG_COLUMNS = (
    "origin_px_x",
    "origin_px_y",
    "initial_direction_deg",
    "user_height_m",
)


def _required_number(
    row: pd.Series, column: str, source: str = "walk_config.csv"
) -> float:
    """必須列を有限な数値へ変換する。"""
    try:
        value = float(row[column])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} の {column} は数値で指定してください。") from exc
    if not np.isfinite(value):
        raise ValueError(f"{source} の {column} は有限値にしてください。")
    return value


def load_measurement_config(
    data_dir: str | Path,
    *,
    image_size_px: tuple[int, int] | None = None,
) -> MeasurementConfig | None:
    """計測CSV、またはAndroid metadataの開始条件を読み取る。"""
    path = Path(data_dir) / "walk_config.csv"
    source = path.name
    if not path.is_file():
        directory = Path(data_dir)
        if not all(
            (directory / name).is_file()
            for name in ("acce.csv", "gyro.csv", "metadata.json")
        ):
            return None
        source = "metadata.json"
        try:
            metadata = json.loads((directory / source).read_text())
            starts = [
                item
                for item in metadata["constraints"]
                if item.get("point_type") == "start"
            ]
            if len(starts) != 1:
                raise ValueError("start制約は1件必要です")
            start = starts[0]
            dataframe = pd.DataFrame(
                [
                    {
                        "origin_px_x": start["x"],
                        "origin_px_y": start["y"],
                        "initial_direction_deg": start["direction"],
                        "user_height_m": None,
                    }
                ]
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError("metadata.jsonのstart制約を読み込めません") from exc
    else:
        try:
            dataframe = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise ValueError(f"walk_config.csv を読み込めません: {path}") from exc
    missing = set(MEASUREMENT_CONFIG_COLUMNS) - set(dataframe.columns)
    if missing:
        raise ValueError(f"walk_config.csv に必須列がありません: {missing}")
    if len(dataframe) != 1:
        raise ValueError("walk_config.csv はヘッダーを除いて1行にしてください。")
    row = dataframe.iloc[0]
    origin_x_value = _required_number(row, "origin_px_x", source)
    origin_y_value = _required_number(row, "origin_px_y", source)
    if not origin_x_value.is_integer() or not origin_y_value.is_integer():
        raise ValueError(f"{source} の起点座標は整数で指定してください。")
    origin = (int(origin_x_value), int(origin_y_value))
    if origin[0] < 0 or origin[1] < 0:
        raise ValueError(f"{source} の起点座標は0以上にしてください。")
    if image_size_px is not None:
        image_width, image_height = image_size_px
        if origin[0] >= image_width or origin[1] >= image_height:
            raise ValueError(
                f"{source} の起点座標がフロアマップ範囲外です: "
                f"{origin} / ({image_width}, {image_height})"
            )
    direction = _required_number(row, "initial_direction_deg", source)
    height_value = row["user_height_m"]
    user_height: float | None = None
    if not pd.isna(height_value) and str(height_value).strip():
        user_height = _required_number(row, "user_height_m")
        if user_height <= 0:
            raise ValueError(
                "walk_config.csv の user_height_m は正の値にしてください。"
            )
    note_value = row.get("note", "")
    note = "" if pd.isna(note_value) else str(note_value).strip()
    return MeasurementConfig(origin, direction, user_height, note)
