"""BLE RSSI をランドマーク検出へ変換する pipeline。

役割:
    BLE CSV の読み込みと RSSI 判定だけを担当し、軌跡補正に依存しない
    共有のランドマーク検出列を作る。
依存元:
    ``ble.lib`` の loader / detection、``common.settings.BleLandmarkSettings``、
    ``common.lib.models.LandmarkDetection`` を使用する。
利用先:
    ``pdr.pipeline.run_pdr`` が通常PDR補正の入力に使用する。将来は
    particle filter が同じ検出列を観測尤度の入力として利用できる。
処理フロー:
    設定が無効なら ``None`` を返し、有効なら CSV 読み込みと閾値判定を
    行って検出列を返す。
"""

from ..common.config import (
    BLE_PATH_LOSS_N,
    BLE_PATH_LOSS_TX_POWER_DBM,
    BLE_RSSI_SIGMA_DB,
)
from ..common.lib.models import LandmarkRange, PathLossModel
from ..common.settings import BleLandmarkSettings
from .lib.detection import detect_landmarks, smoothed_rssi_at_detection
from .lib.loader import load_ble_observations
from .lib.pathloss import rssi_sigma_to_distance_sigma_m, rssi_to_distance_m


def run_ble_landmark_detection(
    settings: BleLandmarkSettings,
) -> tuple[LandmarkRange, ...] | None:
    """BLE ランドマークを検出する。

    ``None`` は機能無効、空タプルは有効だが検出なしを表す。
    """
    if not settings.enabled:
        return None
    observations = load_ble_observations(settings.data_path)
    detections = detect_landmarks(observations, settings)
    landmarks = settings.landmark_map()
    default_model = PathLossModel(
        BLE_PATH_LOSS_TX_POWER_DBM,
        BLE_PATH_LOSS_N,
        BLE_RSSI_SIGMA_DB,
    )
    ranges = []
    for detection in detections:
        landmark = landmarks[detection.beacon_id]
        model = landmark.path_loss_model or default_model
        smoothed_rssi = smoothed_rssi_at_detection(
            observations,
            detection,
            settings.rssi_smoothing_samples,
        )
        distance = rssi_to_distance_m(smoothed_rssi, model)
        ranges.append(
            LandmarkRange(
                detection.timestamp_s,
                detection.beacon_id,
                detection.rssi_dbm,
                distance,
                rssi_sigma_to_distance_sigma_m(distance, model),
                smoothed_rssi,
            )
        )
    return tuple(ranges)
