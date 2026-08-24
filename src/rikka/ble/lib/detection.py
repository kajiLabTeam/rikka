"""RSSI 閾値によるランドマーク検出。

役割:
    BLE 観測列から、ランドマークに十分近づいたと判定できる時刻とビーコンを決める。
依存元:
    ``common.lib.models`` の ``BleObservation`` / ``LandmarkDetection`` と
    ``common.settings.BleLandmarkSettings`` の閾値設定を使用する。
利用先:
    ``ble.pipeline`` が座標補正の前段として呼び出す。座標や軌跡には依存しないため、
    将来 RSSI からの距離推定や確率的な観測モデルへ差し替えられる。
処理フロー:
    同時受信窓ごとにビーコン別の最大 RSSI を比較し、未ラッチ候補のうち最大 RSSI の
    ビーコンをラッチする。ラッチ区間の最大 RSSI を追跡し、解除時または入力終端で
    その最大値の時刻を検出として確定して、時刻順に返す。
"""

from dataclasses import dataclass, field

import numpy as np

from ...common.lib.models import BleObservation, LandmarkDetection
from ...common.settings import BleLandmarkSettings
from .loader import group_by_timestamp


@dataclass
class _LatchState:
    """ラッチ中ビーコンの区間内最大 RSSI と解除進捗を保持する。"""

    best_timestamp_s: float
    best_rssi_dbm: float
    release_streak: int = 0
    observations: list[BleObservation] = field(default_factory=list)


def _finalize_detection(
    beacon_id: str,
    state: _LatchState,
    smoothing_samples: int,
) -> LandmarkDetection:
    """移動中央値が最大の時刻を生RSSI付き検出へ変換する。"""
    if len(state.observations) < smoothing_samples:
        return LandmarkDetection(
            state.best_timestamp_s,
            beacon_id,
            state.best_rssi_dbm,
        )
    raw = np.asarray([item.rssi_dbm for item in state.observations], dtype=float)
    radius = smoothing_samples // 2
    smoothed = np.asarray(
        [
            np.median(raw[max(0, index - radius) : index + radius + 1])
            for index in range(len(raw))
        ],
        dtype=float,
    )
    maximum = float(np.max(smoothed))
    candidates = [
        index for index, value in enumerate(smoothed) if np.isclose(value, maximum)
    ]
    center_timestamp = float(
        np.median([state.observations[index].timestamp_s for index in candidates])
    )
    best_index = min(
        candidates,
        key=lambda index: (
            abs(state.observations[index].timestamp_s - center_timestamp),
            -state.observations[index].rssi_dbm,
            state.observations[index].timestamp_s,
        ),
    )
    best = state.observations[best_index]
    return LandmarkDetection(
        best.timestamp_s,
        beacon_id,
        best.rssi_dbm,
    )


def smoothed_rssi_at_detection(
    observations: tuple[BleObservation, ...],
    detection: LandmarkDetection,
    smoothing_samples: int,
) -> float:
    """検出ビーコンの時系列から検出時刻に対応する移動中央値を返す。"""
    beacon = [item for item in observations if item.beacon_id == detection.beacon_id]
    if not beacon:
        return detection.rssi_dbm
    index = min(
        range(len(beacon)),
        key=lambda item_index: abs(
            beacon[item_index].timestamp_s - detection.timestamp_s
        ),
    )
    radius = smoothing_samples // 2
    values = [
        item.rssi_dbm for item in beacon[max(0, index - radius) : index + radius + 1]
    ]
    return float(np.median(values))


def _select_strongest(candidates: list[BleObservation]) -> BleObservation:
    """最大 RSSI の観測を返す。同値なら beacon_id の辞書順で先のものを返す。"""
    return min(
        candidates,
        key=lambda observation: (-observation.rssi_dbm, observation.beacon_id),
    )


def _reduce_by_beacon(
    group: tuple[BleObservation, ...],
) -> list[BleObservation]:
    """同一窓内の観測をビーコンごとの最大 RSSI へ集約する。"""
    strongest: dict[str, BleObservation] = {}
    for observation in group:
        current = strongest.get(observation.beacon_id)
        if current is None or observation.rssi_dbm > current.rssi_dbm:
            strongest[observation.beacon_id] = observation
    return list(strongest.values())


def detect_landmarks(
    observations: tuple[BleObservation, ...],
    settings: BleLandmarkSettings,
) -> tuple[LandmarkDetection, ...]:
    """RSSI 閾値とラッチ状態からランドマーク検出列を作る。"""
    threshold = settings.rssi_threshold_dbm
    release = threshold - settings.release_margin_db
    known = settings.landmark_map()
    latched: dict[str, _LatchState] = {}
    detections: list[LandmarkDetection] = []

    for _, group in group_by_timestamp(observations, settings.sync_window_s):
        reduced = _reduce_by_beacon(group)
        released: list[str] = []
        for observation in reduced:
            state = latched.get(observation.beacon_id)
            if state is None:
                continue
            state.observations.append(observation)
            if observation.rssi_dbm > state.best_rssi_dbm:
                state.best_timestamp_s = observation.timestamp_s
                state.best_rssi_dbm = observation.rssi_dbm
            if observation.rssi_dbm < release:
                state.release_streak += 1
                if state.release_streak >= settings.release_streak:
                    released.append(observation.beacon_id)
            else:
                state.release_streak = 0
        candidates = [
            observation
            for observation in reduced
            if observation.beacon_id in known
            and observation.beacon_id not in latched
            and observation.rssi_dbm >= threshold
        ]
        if candidates:
            best = _select_strongest(candidates)
            latched[best.beacon_id] = _LatchState(
                best_timestamp_s=best.timestamp_s,
                best_rssi_dbm=best.rssi_dbm,
                observations=[best],
            )
        for beacon_id in released:
            detections.append(
                _finalize_detection(
                    beacon_id,
                    latched.pop(beacon_id),
                    settings.rssi_smoothing_samples,
                )
            )

    detections.extend(
        _finalize_detection(beacon_id, state, settings.rssi_smoothing_samples)
        for beacon_id, state in latched.items()
    )
    detections.sort(key=lambda detection: detection.timestamp_s)

    return tuple(detections)
