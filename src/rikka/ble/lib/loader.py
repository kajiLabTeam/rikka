"""BLE RSSI CSV の読み込みと検証。

役割:
    サンプル・実測を問わず BLE RSSI CSV を読み、時刻昇順の観測列へ正規化する。
依存元:
    ``common.lib.models`` から ``BleObservation`` を取得し、Pandas と NumPy で
    列検証と型変換を行う。
利用先:
    ``ble.pipeline`` がランドマーク補正の入力を用意するときに使用する。
    サンプル生成モジュールとは独立しており、CSV の出所に依存しない。
処理フロー:
    CSV を読み、必須列と値域を検証し、時刻昇順に整列した観測列を返す。
"""

from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.lib.models import BleObservation

BLE_REQUIRED_COLUMNS = ("timestamp_s", "beacon_id", "rssi_dbm")


def load_ble_observations(
    data_path: str | Path,
) -> tuple[BleObservation, ...]:
    """BLE RSSI CSV を読み込み、時刻昇順の観測列を返す。"""
    path = Path(data_path)
    if not path.exists():
        raise ValueError(f"BLE データが存在しません: {path}")
    if not path.is_file():
        raise ValueError(f"BLE データはファイルを指定してください: {path}")
    try:
        dataframe = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"BLE データを CSV として読み込めません: {path}") from exc

    missing = set(BLE_REQUIRED_COLUMNS) - set(dataframe.columns)
    if missing:
        raise ValueError(f"BLE データに必須列がありません: {missing}")
    if dataframe.empty:
        raise ValueError(f"BLE データに観測がありません: {path}")

    timestamps = pd.to_numeric(dataframe["timestamp_s"], errors="coerce")
    if not np.isfinite(timestamps.to_numpy(dtype=float)).all():
        raise ValueError("timestamp_s には有限な数値だけを指定してください。")
    if (timestamps < 0).any():
        raise ValueError("timestamp_s には0以上の値を指定してください。")
    rssi_values = pd.to_numeric(dataframe["rssi_dbm"], errors="coerce")
    if not np.isfinite(rssi_values.to_numpy(dtype=float)).all():
        raise ValueError("rssi_dbm には有限な数値だけを指定してください。")
    beacon_ids = dataframe["beacon_id"].astype(str).str.strip()
    if (beacon_ids == "").any():
        raise ValueError("beacon_id には空でない値を指定してください。")

    dataframe = dataframe.assign(
        timestamp_s=timestamps,
        beacon_id=beacon_ids,
        rssi_dbm=rssi_values,
    ).sort_values("timestamp_s", kind="stable", ignore_index=True)
    return tuple(
        BleObservation(
            timestamp_s=float(row.timestamp_s),
            beacon_id=str(row.beacon_id),
            rssi_dbm=float(row.rssi_dbm),
        )
        for row in dataframe.itertuples(index=False)
    )


def group_by_timestamp(
    observations: tuple[BleObservation, ...],
) -> tuple[tuple[float, tuple[BleObservation, ...]], ...]:
    """同一時刻の観測をまとめた ``(timestamp_s, 観測列)`` の列を返す。"""
    return tuple(
        (timestamp, tuple(group))
        for timestamp, group in groupby(
            observations,
            key=lambda observation: observation.timestamp_s,
        )
    )
