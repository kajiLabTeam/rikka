"""旧解析APIのCSV互換処理を提供する公開窓口。

役割:
    旧CSVの列と時刻基準、任意metadata、座標変換の既存動作を維持する。
依存元:
    analysis_apiの共通実行窓口、common.configの既存既定値、Pandasを使う。
利用先:
    Nozomiの旧/analyzeルートが、通信処理から分離した解析処理として使う。
処理フロー:
    入力を正規化し、共通APIでPDRを実行して、旧形式CSVへ変換する。
"""

import json
from io import BytesIO
from typing import Any, Literal

import pandas as pd

from rikka import analysis_api
from rikka.common.config import FLOORMAP_SCALE, INITIAL_DIRECTION

SensorKind = Literal["acce", "gyro"]

SENSOR_COLUMN_ALIASES: dict[SensorKind, dict[str, tuple[str, ...]]] = {
    "acce": {
        "t": ("t", "Time (s)", "time_s", "time_seconds"),
        "x": ("x", "Acceleration x (m/s^2)", "X (m/s^2)", "x(m/s^2)"),
        "y": ("y", "Acceleration y (m/s^2)", "Y (m/s^2)", "y(m/s^2)"),
        "z": ("z", "Acceleration z (m/s^2)", "Z (m/s^2)", "z(m/s^2)"),
    },
    "gyro": {
        "t": ("t", "Time (s)", "time_s", "time_seconds"),
        "x": ("x", "Gyroscope x (rad/s)", "X (rad/s)", "x(rad/s)"),
        "y": ("y", "Gyroscope y (rad/s)", "Y (rad/s)", "y(rad/s)"),
        "z": ("z", "Gyroscope z (rad/s)", "Z (rad/s)", "z(rad/s)"),
    },
}


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in df), None)


def normalize_sensor_csv(
    df: pd.DataFrame,
    sensor_kind: SensorKind,
    recording_start_wall_time_ms: float | None = None,
) -> pd.DataFrame:
    df = df.rename(columns=lambda column: str(column).strip())
    normalized = pd.DataFrame()

    time_column = _find_column(df, SENSOR_COLUMN_ALIASES[sensor_kind]["t"])
    if time_column is not None:
        normalized["t"] = pd.to_numeric(df[time_column], errors="coerce")
    elif "timestamp_ns" in df.columns:
        timestamp_ns = pd.to_numeric(df["timestamp_ns"], errors="coerce")
        start_offset_s = 0.0
        if recording_start_wall_time_ms is not None and "wall_time_ms" in df.columns:
            first_wall_time_ms = pd.to_numeric(
                df["wall_time_ms"], errors="coerce"
            ).iloc[0]
            start_offset_s = (
                float(first_wall_time_ms) - recording_start_wall_time_ms
            ) / 1_000.0
        normalized["t"] = (
            timestamp_ns - timestamp_ns.iloc[0]
        ) / 1_000_000_000 + start_offset_s
    elif "wall_time_ms" in df.columns:
        wall_time_ms = pd.to_numeric(df["wall_time_ms"], errors="coerce")
        normalized["t"] = (wall_time_ms - wall_time_ms.iloc[0]) / 1_000

    for column_name in ("x", "y", "z"):
        source_column = _find_column(
            df,
            SENSOR_COLUMN_ALIASES[sensor_kind][column_name],
        )
        if source_column is not None:
            normalized[column_name] = pd.to_numeric(
                df[source_column],
                errors="coerce",
            )

    return normalized


def result_timestamps(point_count: int, t_at_steps: list[float]) -> list[float | None]:
    """旧CSVの移動開始を0秒とする時刻列を作る。"""
    if point_count != len(t_at_steps) + 1 or not t_at_steps:
        return [None] * point_count
    return [None, *[float(t - t_at_steps[0]) for t in t_at_steps]]


def build_result_dataframe(
    trajectory: list[list[float]], t_at_steps: list[float]
) -> pd.DataFrame:
    """旧CSVの列順とローカル座標を構成する。"""
    result = pd.DataFrame(trajectory, columns=["rikka_x", "rikka_y"])
    result.insert(0, "step_index", range(len(result)))
    result.insert(1, "rikka_timestamp_s", result_timestamps(len(result), t_at_steps))
    result["x"] = result["rikka_x"]
    result["y"] = result["rikka_y"]
    return result


def analyze_to_csv(inputs: dict[str, bytes], parameters: dict[str, Any]) -> bytes:
    """任意metadataを受け取り、共通実行APIの結果を旧CSVへ変換する。"""
    recording_start = None
    if "metadata" in inputs:
        metadata = json.loads(inputs["metadata"])
        value = metadata.get("recording_start_wall_time_ms")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("metadata.json の recording_start_wall_time_ms が不正です")
        recording_start = float(value)
    normalized_inputs = {}
    for kind in ("acce", "gyro"):
        frame = normalize_sensor_csv(
            pd.read_csv(BytesIO(inputs[kind])), kind, recording_start
        )
        missing = {"x", "y", "z"} - set(frame.columns)
        if missing:
            raise ValueError(
                f"{kind} csv missing required columns: {', '.join(sorted(missing))}"
            )
        normalized_inputs[kind] = frame.to_csv(index=False).encode()
    # metadataを持たない旧入力の時刻基準は正規化時に確定済み。
    normalized_inputs["metadata"] = b'{"recording_start_wall_time_ms":0}'
    direction = parameters.get("initial_direction")
    result = analysis_api.execute_method(
        "pdr",
        normalized_inputs,
        {
            "initial_direction": float(
                INITIAL_DIRECTION if direction is None else direction
            )
        },
    )["trajectory"]
    points = result["points"]
    timestamps = [point["timestamp_s"] for point in points[1:]]
    step_times = (
        [float(t) for t in timestamps]
        if all(isinstance(t, (int, float)) for t in timestamps)
        else []
    )
    frame = build_result_dataframe(
        [[point["x"], point["y"]] for point in points], step_times
    )
    if (
        parameters.get("origin_x") is not None
        and parameters.get("origin_y") is not None
    ):
        scale = parameters.get("floor_scale")
        scale = float(FLOORMAP_SCALE if scale is None else scale)
        frame["x"] = float(parameters["origin_x"]) + frame["rikka_x"] / scale
        frame["y"] = float(parameters["origin_y"]) + frame["rikka_y"] / scale
    return str(frame.to_csv(index=False)).encode()
