"""サンプル BLE RSSI データの生成。

役割:
    実測 BLE データが無い段階で、既存の歩行データと同じ時間軸を持つサンプル
    RSSI 列を作り、CSV として書き出す。
依存元:
    ``common.settings.BleSampleSettings`` から生成条件、``common.lib.models`` から
    ``BleObservation`` を取得し、NumPy と Pandas で信号生成と保存を行う。
利用先:
    CLI の ``ble-sample`` コマンドだけが使用する。本番のランドマーク測位処理
    （loader / detection / correction）はこのモジュールを import しない。
処理フロー:
    センサー時刻範囲からアドバタイズ時刻を作り、ビーコンごとにガウス形状の RSSI と
    ノイズを重ね、下限でクリップして時刻昇順の観測列と CSV を生成する。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ...common.lib.models import BleObservation
from ...common.settings import BleSampleSettings


def build_sample_times(
    df_acc: pd.DataFrame,
    settings: BleSampleSettings,
) -> np.ndarray:
    """センサー時刻範囲から BLE アドバタイズ時刻の列を作る。"""
    if "t" not in df_acc.columns:
        raise ValueError("サンプル BLE の時間同期には加速度データの t 列が必要です。")
    t_start = float(df_acc["t"].min())
    t_end = float(df_acc["t"].max())
    if not np.isfinite(t_start) or not np.isfinite(t_end):
        raise ValueError("加速度データの t 列には有限な時刻が必要です。")
    times = np.arange(t_start, t_end + 1e-9, settings.interval_s)
    if len(times) == 0:
        raise ValueError("サンプル BLE の生成対象時刻がありません。")
    return times


def generate_sample_observations(
    times: np.ndarray,
    settings: BleSampleSettings,
) -> tuple[BleObservation, ...]:
    """時刻列とビーコン定義からサンプル RSSI 観測列を作る。"""
    rng = np.random.default_rng(settings.seed)
    span = settings.peak_rssi_dbm - settings.base_rssi_dbm
    per_beacon: list[tuple[str, np.ndarray]] = []
    for beacon_id, peak_time in settings.peak_times_s:
        shape = np.exp(-((times - peak_time) ** 2) / (2.0 * settings.sigma_s**2))
        noise = rng.normal(0.0, settings.noise_sigma_db, size=len(times))
        rssi = np.maximum(
            settings.base_rssi_dbm + span * shape + noise,
            settings.min_rssi_dbm,
        )
        per_beacon.append((beacon_id, rssi))

    return tuple(
        BleObservation(float(timestamp), beacon_id, round(float(rssi[index]), 2))
        for index, timestamp in enumerate(times)
        for beacon_id, rssi in per_beacon
    )


def write_sample_csv(
    observations: tuple[BleObservation, ...],
    output_path: str | Path,
) -> Path:
    """サンプル観測列を BLE CSV として保存し、保存先を返す。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(
        (
            {
                "timestamp_s": round(item.timestamp_s, 3),
                "beacon_id": item.beacon_id,
                "rssi_dbm": item.rssi_dbm,
            }
            for item in observations
        ),
        columns=["timestamp_s", "beacon_id", "rssi_dbm"],
    )
    dataframe.to_csv(path, index=False)
    return path


def generate_sample_csv(
    df_acc: pd.DataFrame,
    output_path: str | Path,
    settings: BleSampleSettings | None = None,
) -> tuple[Path, int]:
    """センサーデータからサンプル BLE CSV を生成し、保存先と行数を返す。"""
    resolved_settings = BleSampleSettings() if settings is None else settings
    times = build_sample_times(df_acc, resolved_settings)
    observations = generate_sample_observations(times, resolved_settings)
    return write_sample_csv(observations, output_path), len(observations)
