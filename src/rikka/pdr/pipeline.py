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

import matplotlib.image as mpimg
import pandas as pd

from ..ble.lib.ranging_check import run_ranging_preflight
from ..ble.pipeline import run_ble_landmark_detection
from ..common.lib.floormap import normalize_floormap_gray
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
        consistency = run_ranging_preflight(
            settings.landmark,
            floormap,
            prepared.trajectory,
            prepared.t_at_steps,
            prepared.gx_mean,
            prepared.gz_mean,
        )
        landmark = apply_landmark_corrections(
            prepared.trajectory,
            prepared.t_at_steps,
            step_headings=prepared.step_headings,
            step_lengths=prepared.step_lengths,
            detections=detections,
            landmarks=settings.landmark.landmarks,
            floormap=floormap,
            data_path=str(settings.landmark.data_path),
            rssi_threshold_dbm=settings.landmark.rssi_threshold_dbm,
            gx_mean=prepared.gx_mean,
            gz_mean=prepared.gz_mean,
            correction_mode=settings.landmark.correction_mode,
            max_correction_m=settings.landmark.max_correction_m,
            max_warp_span_m=settings.landmark.max_warp_span_m,
            retrofit_forward_mode=settings.landmark.retrofit_forward_mode,
            retrofit_max_heading_deg=settings.landmark.retrofit_max_heading_deg,
            retrofit_stride_scale_min=(settings.landmark.retrofit_stride_scale_min),
            retrofit_stride_scale_max=(settings.landmark.retrofit_stride_scale_max),
            retrofit_min_span_m=settings.landmark.retrofit_min_span_m,
            retrofit_map_check=settings.landmark.retrofit_map_check,
            retrofit_damp_factors=settings.landmark.retrofit_damp_factors,
            map_gray=(
                normalize_floormap_gray(mpimg.imread(floormap.path))
                if settings.landmark.correction_mode == "similarity"
                and settings.landmark.retrofit_map_check != "off"
                else None
            ),
            ranging_consistency=consistency,
        )
    return TrajectoryResult(
        trajectory=prepared.trajectory if landmark is None else landmark.trajectory,
        step_lengths=(
            prepared.step_lengths
            if landmark is None or landmark.step_lengths is None
            else landmark.step_lengths
        ),
        t_at_steps=prepared.t_at_steps,
        step_headings=(
            prepared.step_headings
            if landmark is None or landmark.step_headings is None
            else landmark.step_headings
        ),
        prepared=prepared,
        landmark=landmark,
    )
