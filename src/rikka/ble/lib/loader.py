"""BLE RSSI CSV の読み込みと検証。

役割:
    標準形式またはThingsup形式の BLE RSSI CSV と端末座標CSVを読み、時刻昇順の
    観測列と既知ランドマークへ正規化する。
依存元:
    ``common.lib.models`` から ``BleObservation`` と ``Landmark`` を取得し、
    Pandas、NumPy、phyphox の ``meta/time.csv`` / ``Time Reference.csv`` で
    列検証と時刻変換を行う。
利用先:
    ``ble.pipeline`` がランドマーク補正の入力を用意するときに使用する。
    CLI は計測ディレクトリの ``BLE_pos.csv`` から既知座標を構築するためにも使う。
処理フロー:
    CSV形式を判定し、Thingsup形式では端末照合と実験開始基準への時刻変換を行い、
    必須列と値域を検証して時刻昇順に整列した観測列を返す。
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ...common.lib.models import BleObservation, Landmark

BLE_REQUIRED_COLUMNS = ("timestamp_s", "beacon_id", "rssi_dbm")
BLE_LOGGER_REQUIRED_COLUMNS = (
    "Timestamp",
    "Device Name",
    "MAC Address",
    "RSSI",
    "Raw Data",
)
BLE_POSITION_REQUIRED_COLUMNS = (
    "beacon_id",
    "device_name",
    "mac_address",
    "raw_data_suffix",
    "pixel_x",
    "pixel_y",
)
BLE_TIME_REFERENCE_COLUMN = "Unix Offset (s)"
PHYPOX_META_TIME_COLUMNS = ("event", "experiment time", "system time")
BLE_LOGGER_TIMEZONE = ZoneInfo("Asia/Tokyo")
BLE_LOGGER_TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S:%f"


@dataclass(frozen=True)
class _BlePositionRecord:
    """BLEログの端末情報とRikka内の識別子・座標を保持する。"""

    beacon_id: str
    device_name: str
    mac_address: str
    raw_data_suffix: str
    pixel_x: float | None
    pixel_y: float | None


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    """パスを検証してCSVを読み、読み込み失敗を統一した例外に変換する。"""
    if not path.exists():
        raise ValueError(f"{label}が存在しません: {path}")
    if not path.is_file():
        raise ValueError(f"{label}はファイルを指定してください: {path}")
    try:
        return pd.read_csv(path)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError(f"{label}を CSV として読み込めません: {path}") from exc


def _required_columns(
    dataframe: pd.DataFrame,
    required: tuple[str, ...],
    label: str,
) -> None:
    """CSVの必須列が揃っていることを検証する。"""
    missing = set(required) - set(dataframe.columns)
    if missing:
        raise ValueError(f"{label}に必須列がありません: {missing}")


def _normalize_mac_address(value: object) -> str:
    """MACアドレスを区切り文字に依存しない比較用文字列へ揃える。"""
    return "".join(character for character in str(value).upper() if character.isalnum())


def _normalize_raw_data(value: object) -> str:
    """Raw Dataを小文字・接頭辞なしの16進文字列へ揃える。"""
    normalized = str(value).strip().lower()
    return normalized[2:] if normalized.startswith("0x") else normalized


def _optional_coordinate(value: object, column: str, row_number: int) -> float | None:
    """空欄を未設置として許容し、入力済み座標だけ有限値へ変換する。"""
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        coordinate = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"BLE_pos.csv の {row_number} 行目の {column} は数値で指定してください。"
        ) from exc
    if not np.isfinite(coordinate):
        raise ValueError(
            f"BLE_pos.csv の {row_number} 行目の {column} は有限値で指定してください。"
        )
    return coordinate


def _load_position_records(data_path: str | Path) -> tuple[_BlePositionRecord, ...]:
    """BLE_pos.csvから端末照合情報と任意の既知座標を読み込む。"""
    path = Path(data_path)
    dataframe = _read_csv(path, "BLE 端末座標データ")
    _required_columns(dataframe, BLE_POSITION_REQUIRED_COLUMNS, "BLE_pos.csv ")
    if dataframe.empty:
        raise ValueError(f"BLE_pos.csv に端末情報がありません: {path}")

    records: list[_BlePositionRecord] = []
    beacon_ids: set[str] = set()
    for row_number, row in enumerate(dataframe.itertuples(index=False), start=2):
        values = row._asdict()
        beacon_id = str(values["beacon_id"]).strip()
        device_name = str(values["device_name"]).strip()
        mac_address = _normalize_mac_address(values["mac_address"])
        raw_data_suffix = _normalize_raw_data(values["raw_data_suffix"])
        if not all((beacon_id, device_name, mac_address, raw_data_suffix)):
            raise ValueError(
                f"BLE_pos.csv の {row_number} 行目に空の端末識別子があります。"
            )
        if beacon_id in beacon_ids:
            raise ValueError(f"BLE_pos.csv の beacon_id が重複しています: {beacon_id}")
        beacon_ids.add(beacon_id)
        pixel_x = _optional_coordinate(values["pixel_x"], "pixel_x", row_number)
        pixel_y = _optional_coordinate(values["pixel_y"], "pixel_y", row_number)
        if (pixel_x is None) != (pixel_y is None):
            raise ValueError(
                f"BLE_pos.csv の {row_number} 行目は pixel_x と pixel_y を"
                "両方入力するか、両方空欄にしてください。"
            )
        records.append(
            _BlePositionRecord(
                beacon_id,
                device_name,
                mac_address,
                raw_data_suffix,
                pixel_x,
                pixel_y,
            )
        )
    return tuple(records)


def load_ble_landmarks(data_path: str | Path) -> tuple[Landmark, ...]:
    """BLE_pos.csvから座標が確定しているランドマークだけを返す。"""
    return tuple(
        Landmark(record.beacon_id, record.pixel_x, record.pixel_y)
        for record in _load_position_records(data_path)
        if record.pixel_x is not None and record.pixel_y is not None
    )


def _load_unix_offset(path: Path) -> float:
    """phyphoxのTime Reference.csvから実験時刻のUnixオフセットを読む。"""
    dataframe = _read_csv(path, "Time Reference.csv ")
    _required_columns(dataframe, (BLE_TIME_REFERENCE_COLUMN,), "Time Reference.csv ")
    values = pd.to_numeric(dataframe[BLE_TIME_REFERENCE_COLUMN], errors="coerce")
    if len(values) != 1 or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(
            "Time Reference.csv の Unix Offset (s) は有限な数値1件にしてください。"
        )
    return float(values.iloc[0])


def _load_experiment_unix_offset(data_path: Path) -> float:
    """高精度なphyphoxメタ時刻を優先し、なければTime Referenceへ戻る。"""
    meta_time_path = data_path.parent / "meta" / "time.csv"
    if not meta_time_path.is_file():
        return _load_unix_offset(data_path.with_name("Time Reference.csv"))

    dataframe = _read_csv(meta_time_path, "phyphox meta/time.csv ")
    _required_columns(dataframe, PHYPOX_META_TIME_COLUMNS, "phyphox meta/time.csv ")
    start_rows = dataframe.loc[dataframe["event"].astype(str).str.strip() == "START"]
    if len(start_rows) != 1:
        raise ValueError("phyphox meta/time.csv の START は1件にしてください。")
    experiment_time = pd.to_numeric(
        start_rows["experiment time"], errors="coerce"
    ).iloc[0]
    system_time = pd.to_numeric(start_rows["system time"], errors="coerce").iloc[0]
    if not np.isfinite(experiment_time) or not np.isfinite(system_time):
        raise ValueError(
            "phyphox meta/time.csv の START 時刻は有限な数値にしてください。"
        )
    return float(system_time - experiment_time)


def _match_beacon_id(
    row: dict[str, object],
    records: tuple[_BlePositionRecord, ...],
) -> str | None:
    """Device Name・MAC・Raw Data末尾がすべて一致する端末を特定する。"""
    device_name = str(row["Device Name"]).strip()
    mac_address = _normalize_mac_address(row["MAC Address"])
    raw_data = _normalize_raw_data(row["Raw Data"])
    matches = [
        record.beacon_id
        for record in records
        if record.device_name == device_name
        and record.mac_address == mac_address
        and raw_data.endswith(record.raw_data_suffix)
    ]
    if len(matches) > 1:
        raise ValueError(
            f"BLEログの端末情報が複数の beacon_id に一致しました: {matches}"
        )
    return matches[0] if matches else None


def _load_logger_observations(
    dataframe: pd.DataFrame,
    data_path: Path,
) -> tuple[BleObservation, ...]:
    """Thingsup形式のBLEログをphyphox開始基準の観測列へ変換する。"""
    _required_columns(dataframe, BLE_LOGGER_REQUIRED_COLUMNS, "BLE データ")
    position_path = data_path.with_name("BLE_pos.csv")
    records = _load_position_records(position_path)
    unix_offset_s = _load_experiment_unix_offset(data_path)

    observations: list[BleObservation] = []
    for row_number, row in enumerate(dataframe.to_dict("records"), start=2):
        beacon_id = _match_beacon_id(row, records)
        if beacon_id is None:
            continue
        try:
            timestamp = datetime.strptime(
                str(row["Timestamp"]).strip(), BLE_LOGGER_TIMESTAMP_FORMAT
            ).replace(tzinfo=BLE_LOGGER_TIMEZONE)
        except ValueError as exc:
            raise ValueError(
                f"BLE.csv の {row_number} 行目の Timestamp 形式が不正です。"
            ) from exc
        timestamp_s = timestamp.timestamp() - unix_offset_s
        try:
            rssi_dbm = float(row["RSSI"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"BLE.csv の {row_number} 行目の RSSI は数値で指定してください。"
            ) from exc
        if not np.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError(
                f"BLE.csv の {row_number} 行目は実験開始前の時刻です: "
                f"{timestamp_s:.3f} s"
            )
        if not np.isfinite(rssi_dbm):
            raise ValueError(
                f"BLE.csv の {row_number} 行目の RSSI は有限値で指定してください。"
            )
        if rssi_dbm == 127:
            continue
        observations.append(BleObservation(timestamp_s, beacon_id, rssi_dbm))
    if not observations:
        raise ValueError(f"BLE データに照合可能な有効観測がありません: {data_path}")
    return tuple(sorted(observations, key=lambda item: item.timestamp_s))


def load_ble_observations(
    data_path: str | Path,
) -> tuple[BleObservation, ...]:
    """BLE RSSI CSV を読み込み、時刻昇順の観測列を返す。"""
    path = Path(data_path)
    dataframe = _read_csv(path, "BLE データ")
    if set(BLE_LOGGER_REQUIRED_COLUMNS).issubset(dataframe.columns):
        return _load_logger_observations(dataframe, path)

    _required_columns(dataframe, BLE_REQUIRED_COLUMNS, "BLE データ")
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
    window_s: float = 0.0,
) -> tuple[tuple[float, tuple[BleObservation, ...]], ...]:
    """許容窓に収まる観測を同時受信としてまとめた列を返す。

    ``window_s`` が 0 のときは時刻の完全一致でまとめる。実測 BLE のように
    ビーコンごとに受信時刻がずれる場合は、窓幅を与えて同時受信として扱う。
    """
    groups: list[tuple[float, list[BleObservation]]] = []
    for observation in observations:
        if groups and observation.timestamp_s - groups[-1][0] <= window_s:
            groups[-1][1].append(observation)
            continue
        groups.append((observation.timestamp_s, [observation]))
    return tuple((timestamp, tuple(items)) for timestamp, items in groups)
