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
    同一時刻ごとに登録済みビーコンの RSSI を比較し、ラッチ状態を考慮して最大 RSSI の
    ビーコンを 1 件検出し、RSSI が解除水準を2観測連続で下回るとラッチを解除する。
"""

from ...common.lib.models import BleObservation, LandmarkDetection
from ...common.settings import BleLandmarkSettings
from .loader import group_by_timestamp


def _select_strongest(candidates: list[BleObservation]) -> BleObservation:
    """最大 RSSI の観測を返す。同値なら beacon_id の辞書順で先のものを返す。"""
    return min(
        candidates,
        key=lambda observation: (-observation.rssi_dbm, observation.beacon_id),
    )


def detect_landmarks(
    observations: tuple[BleObservation, ...],
    settings: BleLandmarkSettings,
) -> tuple[LandmarkDetection, ...]:
    """RSSI 閾値とラッチ状態からランドマーク検出列を作る。"""
    threshold = settings.rssi_threshold_dbm
    release = threshold - settings.release_margin_db
    known = settings.landmark_map()
    latched: set[str] = set()
    release_streaks: dict[str, int] = {}
    detections: list[LandmarkDetection] = []

    for timestamp, group in group_by_timestamp(observations):
        released: set[str] = set()
        for observation in group:
            if observation.beacon_id not in latched:
                continue
            if observation.rssi_dbm < release:
                streak = release_streaks.get(observation.beacon_id, 0) + 1
                release_streaks[observation.beacon_id] = streak
                if streak >= 2:
                    released.add(observation.beacon_id)
            else:
                release_streaks[observation.beacon_id] = 0
        candidates = [
            observation
            for observation in group
            if observation.beacon_id in known
            and observation.beacon_id not in latched
            and observation.rssi_dbm >= threshold
        ]
        if candidates:
            best = _select_strongest(candidates)
            detections.append(
                LandmarkDetection(timestamp, best.beacon_id, best.rssi_dbm)
            )
            latched.add(best.beacon_id)
            release_streaks[best.beacon_id] = 0
        latched -= released
        for beacon_id in released:
            release_streaks.pop(beacon_id, None)

    return tuple(detections)
