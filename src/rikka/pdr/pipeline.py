"""センサー入力を通常PDR結果へ変換する pipeline。

役割:
    入力データを取得し、検証済み設定で共有歩列を準備して BLE 補正後の結果へまとめる。
依存元:
    common のセンサー入力・設定・結果型、``lib.preparation``、
    ``ble.pipeline`` の検出列、``lib.landmark_correction`` のPDR補正を利用する。
利用先:
    CLI が最初に呼び、必要なら同じ準備済み歩列を particle pipeline へ渡す。
処理フロー:
    入力ペア確認、既定CSV読み込み、共有歩列準備、BLE補正、結果型構築の順に処理する。
"""

import pandas as pd

from ..ble.pipeline import run_ble_landmark_detection
from ..common.lib.models import FloorMap, TrajectoryResult
from ..common.lib.sensors import load_sensor_data
from ..common.settings import PdrSettings
from .lib.landmark_correction import apply_landmark_corrections
from .lib.preparation import prepare_pdr_steps_with_settings


def run_pdr(
    settings: PdrSettings,
    df_acc: pd.DataFrame | None = None,
    df_gyro: pd.DataFrame | None = None,
    floormap: FloorMap | None = None,
) -> TrajectoryResult:
    """センサー入力を共有ステップと通常PDR軌跡へ変換する。"""
    if settings.landmark.enabled and floormap is None:
        raise ValueError("BLE ランドマーク補正を有効にする場合は floormap が必要です。")
    if (df_acc is None) != (df_gyro is None):
        raise ValueError("df_acc と df_gyro は両方渡すか、両方省略してください。")
    if df_acc is None and df_gyro is None:
        df_acc, df_gyro = load_sensor_data()
    if df_acc is None or df_gyro is None:
        raise RuntimeError("内部エラー: センサーデータが取得できませんでした。")

    prepared = prepare_pdr_steps_with_settings(df_acc, df_gyro, settings)
    detections = run_ble_landmark_detection(settings.landmark)
    landmark = None
    if detections is not None:
        if floormap is None:
            raise RuntimeError("内部エラー: ランドマーク用 floormap がありません。")
        landmark = apply_landmark_corrections(
            prepared.trajectory,
            prepared.t_at_steps,
            detections=detections,
            landmarks=settings.landmark.landmarks,
            floormap=floormap,
            data_path=str(settings.landmark.data_path),
            rssi_threshold_dbm=settings.landmark.rssi_threshold_dbm,
            gx_mean=prepared.gx_mean,
            gz_mean=prepared.gz_mean,
        )
    return TrajectoryResult(
        trajectory=prepared.trajectory if landmark is None else landmark.trajectory,
        step_lengths=prepared.step_lengths,
        t_at_steps=prepared.t_at_steps,
        step_headings=prepared.step_headings,
        prepared=prepared,
        landmark=landmark,
    )
