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

from dataclasses import replace

from ..common.config import (
    BLE_PATH_LOSS_N,
    BLE_PATH_LOSS_TX_POWER_DBM,
    BLE_RSSI_SIGMA_DB,
)
from ..common.lib.models import (
    BleObservation,
    BleRangingInput,
    LandmarkDetection,
    LandmarkRange,
    PathLossModel,
)
from ..common.settings import BleLandmarkSettings
from .lib.detection import detect_landmarks, smoothed_rssi_at_detection
from .lib.loader import is_logger_ble_data, load_ble_observations
from .lib.pathloss import rssi_sigma_to_distance_sigma_m, rssi_to_distance_m


def run_ble_landmark_detection(
    settings: BleLandmarkSettings,
) -> tuple[LandmarkRange, ...] | None:
    """BLE ランドマークを検出する。

    ``None`` は機能無効、空タプルは有効だが検出なしを表す。
    """
    if not settings.enabled:
        return None
    ranging_input = run_ble_ranging_input(settings)
    return None if ranging_input is None else ranging_input.detections


def _to_range(
    observation: BleObservation | LandmarkDetection,
    observations: tuple[BleObservation, ...],
    settings: BleLandmarkSettings,
) -> LandmarkRange:
    """1観測へ設定済みパスロスモデルの距離情報を付与する。"""
    landmarks = settings.landmark_map()
    default_model = PathLossModel(
        BLE_PATH_LOSS_TX_POWER_DBM,
        BLE_PATH_LOSS_N,
        BLE_RSSI_SIGMA_DB,
    )
    landmark = landmarks[observation.beacon_id]
    model = landmark.path_loss_model or default_model
    smoothed_rssi = smoothed_rssi_at_detection(
        observations,
        LandmarkDetection(
            observation.timestamp_s, observation.beacon_id, observation.rssi_dbm
        ),
        settings.rssi_smoothing_samples,
    )
    distance = rssi_to_distance_m(smoothed_rssi, model)
    return LandmarkRange(
        observation.timestamp_s,
        observation.beacon_id,
        observation.rssi_dbm,
        distance,
        rssi_sigma_to_distance_sigma_m(distance, model),
        smoothed_rssi,
    )


def run_ble_ranging_input(settings: BleLandmarkSettings) -> BleRangingInput | None:
    """検出イベントと座標確定ビーコンの全測距観測を分離して返す。"""
    if not settings.enabled:
        return None
    observations = load_ble_observations(settings.data_path)
    known = settings.landmark_map()
    detection_settings = (
        settings
        if is_logger_ble_data(settings.data_path)
        else replace(
            settings,
            detect_min_samples=1,
            detect_cooldown_s=0.0,
            detect_min_prominence_db=0.0,
        )
    )
    detections = detect_landmarks(observations, detection_settings)
    return BleRangingInput(
        detections=tuple(
            _to_range(item, observations, settings) for item in detections
        ),
        observations=tuple(
            _to_range(item, observations, settings)
            for item in observations
            if item.beacon_id in known
        ),
    )
