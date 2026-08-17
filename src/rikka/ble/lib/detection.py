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

from dataclasses import dataclass

from ...common.lib.models import BleObservation, LandmarkDetection
from ...common.settings import BleLandmarkSettings
from .loader import group_by_timestamp


@dataclass
class _LatchState:
    """ラッチ中ビーコンの区間内最大 RSSI と解除進捗を保持する。"""

    best_timestamp_s: float
    best_rssi_dbm: float
    release_streak: int = 0


def _finalize_detection(
    beacon_id: str,
    state: _LatchState,
) -> LandmarkDetection:
    """ラッチ区間の最大 RSSI 観測をランドマーク検出へ変換する。"""
    return LandmarkDetection(
        state.best_timestamp_s,
        beacon_id,
        state.best_rssi_dbm,
    )


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
            )
        for beacon_id in released:
            detections.append(_finalize_detection(beacon_id, latched.pop(beacon_id)))

    detections.extend(
        _finalize_detection(beacon_id, state) for beacon_id, state in latched.items()
    )
    detections.sort(key=lambda detection: detection.timestamp_s)

    return tuple(detections)
