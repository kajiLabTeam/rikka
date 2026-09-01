"""滞在分析入力に共通する検証処理。

役割:
    trajectory列、数値の有限性、parameter精度、floor情報を防御的に検証する。
依存元:
    pandas DataFrameとOkarinで正規化された数値を受け取る。
利用先:
    analysis配下のenrichment、grid、artifactが計算前の共通検証に使用する。
処理フロー:
    型・範囲・欠損・時系列規則を確認し、不正時はValueErrorを送出する。
"""

import math
from decimal import Decimal, InvalidOperation
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

REQUIRED_COLUMNS = (
    "step_index",
    "rikka_timestamp_s",
    "rikka_x",
    "rikka_y",
    "x",
    "y",
)


def validate_parameter(
    value: float,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{name} must be a finite decimal") from error
    exponent = decimal.as_tuple().exponent
    if not isinstance(exponent, int) or exponent < -3:
        raise ValueError(f"{name} must have at most 3 decimal places")
    return number


def require_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def finite_numeric(dataframe: pd.DataFrame, column: str) -> NDArray[np.float64]:
    series = dataframe[column]
    if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
        raise ValueError(f"{column} must be numeric")
    values = cast(NDArray[np.float64], series.to_numpy(dtype=float))
    if not np.isfinite(values).all():
        raise ValueError(f"{column} must contain only finite values")
    return values


def validate_trajectory(dataframe: pd.DataFrame) -> None:
    require_columns(dataframe, REQUIRED_COLUMNS)
    if dataframe.empty:
        raise ValueError("trajectory must contain at least one row")

    steps = dataframe["step_index"]
    if not is_integer_dtype(steps.dtype) or is_bool_dtype(steps.dtype):
        raise ValueError("step_index must be integer")
    if steps.tolist() != list(range(len(dataframe))):
        raise ValueError("step_index must start at 0 and contain no gaps")

    for column in ("rikka_x", "rikka_y", "x", "y"):
        finite_numeric(dataframe, column)

    timestamps = dataframe["rikka_timestamp_s"]
    if not is_numeric_dtype(timestamps.dtype):
        raise ValueError("rikka_timestamp_s must be numeric")
    if not pd.isna(timestamps.iloc[0]):
        raise ValueError("first rikka_timestamp_s must be null")
    if len(timestamps) == 1:
        return

    remaining = timestamps.iloc[1:].to_numpy(dtype=float)
    if not np.isfinite(remaining).all():
        raise ValueError("rikka_timestamp_s contains an invalid value")
    if remaining[0] != 0.0:
        raise ValueError("second rikka_timestamp_s must be 0.0")
    if len(remaining) > 1 and not (np.diff(remaining) > 0).all():
        raise ValueError("rikka_timestamp_s must be strictly increasing")


def validate_floor(
    *,
    map_width_px: int,
    map_height_px: int,
    floor_scale: float,
) -> float:
    if isinstance(map_width_px, bool) or not isinstance(map_width_px, int):
        raise ValueError("map_width_px must be a positive integer")
    if isinstance(map_height_px, bool) or not isinstance(map_height_px, int):
        raise ValueError("map_height_px must be a positive integer")
    if map_width_px <= 0 or map_height_px <= 0:
        raise ValueError("map dimensions must be positive")
    scale = float(floor_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("floor_scale must be positive and finite")
    return scale


def validate_point(value: float, *, name: str, limit: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    point = float(value)
    if not math.isfinite(point) or not 0 <= point < limit:
        raise ValueError(f"{name} must be inside the floor map")
    return point
