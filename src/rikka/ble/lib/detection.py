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
    above_threshold_count: int = 0
    current_above_threshold_count: int = 0


def _finalize_detection(
    beacon_id: str,
    state: _LatchState,
    smoothing_samples: int,
    all_observations: tuple[BleObservation, ...],
    min_samples: int,
    min_prominence_db: float,
    threshold_dbm: float,
) -> LandmarkDetection | None:
    """移動中央値が最大の時刻を生RSSI付き検出へ変換する。"""
    if state.above_threshold_count < min_samples:
        return None
    legacy_peak_selection = min_samples == 1 and min_prominence_db == 0.0
    if legacy_peak_selection:
        if len(state.observations) < smoothing_samples:
            return LandmarkDetection(
                state.best_timestamp_s,
                beacon_id,
                state.best_rssi_dbm,
            )
        window = state.observations
    else:
        beacon = [item for item in all_observations if item.beacon_id == beacon_id]
        radius = smoothing_samples // 2
        start = min(
            range(len(beacon)),
            key=lambda index: abs(
                beacon[index].timestamp_s - state.observations[0].timestamp_s
            ),
        )
        end = min(
            range(len(beacon)),
            key=lambda index: abs(
                beacon[index].timestamp_s - state.observations[-1].timestamp_s
            ),
        )
        if start < radius or end + radius >= len(beacon):
            return None
        window = beacon[max(0, start - radius) : min(len(beacon), end + radius + 1)]
    radius = smoothing_samples // 2
    raw = np.asarray([item.rssi_dbm for item in window], dtype=float)
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
        np.median([window[index].timestamp_s for index in candidates])
    )
    best_index = min(
        candidates,
        key=lambda index: (
            abs(window[index].timestamp_s - center_timestamp),
            -window[index].rssi_dbm,
            window[index].timestamp_s,
        ),
    )
    if (
        maximum - float(np.min(smoothed)) < min_prominence_db
        or maximum - threshold_dbm < min_prominence_db
    ):
        return None
    best = window[best_index]
    if legacy_peak_selection:
        return LandmarkDetection(best.timestamp_s, beacon_id, best.rssi_dbm)
    peak_start = window[candidates[0]].timestamp_s
    peak_end = window[candidates[-1]].timestamp_s
    midpoint = (peak_start + peak_end) / 2.0
    timestamp = max(
        window[index].timestamp_s
        for index in candidates
        if window[index].timestamp_s <= midpoint
    )
    return LandmarkDetection(
        timestamp,
        beacon_id,
        max(window[index].rssi_dbm for index in candidates),
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
            if observation.rssi_dbm >= threshold:
                state.current_above_threshold_count += 1
                state.above_threshold_count = max(
                    state.above_threshold_count,
                    state.current_above_threshold_count,
                )
            else:
                state.current_above_threshold_count = 0
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
                above_threshold_count=1,
                current_above_threshold_count=1,
            )
        for beacon_id in released:
            detection = _finalize_detection(
                beacon_id,
                latched.pop(beacon_id),
                settings.rssi_smoothing_samples,
                observations,
                settings.detect_min_samples,
                settings.detect_min_prominence_db,
                threshold,
            )
            if detection is not None:
                detections.append(detection)

    for beacon_id, state in latched.items():
        detection = _finalize_detection(
            beacon_id,
            state,
            settings.rssi_smoothing_samples,
            observations,
            settings.detect_min_samples,
            settings.detect_min_prominence_db,
            threshold,
        )
        if detection is not None:
            detections.append(detection)
    detections.sort(key=lambda detection: detection.timestamp_s)
    filtered: list[LandmarkDetection] = []
    for detection in detections:
        conflicting = [
            (index, previous)
            for index, previous in enumerate(filtered)
            if previous.beacon_id == detection.beacon_id
            and detection.timestamp_s - previous.timestamp_s
            < settings.detect_cooldown_s
        ]
        if not conflicting:
            filtered.append(detection)
            continue
        index, previous = conflicting[-1]
        if detection.rssi_dbm > previous.rssi_dbm:
            filtered[index] = detection
    filtered.sort(key=lambda detection: detection.timestamp_s)
    return tuple(filtered)
