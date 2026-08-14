"""BLE 観測を軌跡補正へ変換する pipeline。

役割:
    BLE CSV の読み込み、ランドマーク検出、座標補正を順に実行し、補正済み軌跡へ
    まとめる。
依存元:
    ``ble.lib`` の loader / detection / correction と ``common.settings`` の
    ``BleLandmarkSettings`` を使用する。
利用先:
    ``pdr.pipeline.run_pdr`` が BLE 補正を有効にしたときに呼び出す。
処理フロー:
    設定が無効なら何もせず、有効なら CSV 読み込み、閾値判定、逐次補正の順に処理する。
"""

from ..common.lib.models import LandmarkCorrectionResult
from ..common.settings import BleLandmarkSettings
from .lib.correction import apply_landmark_corrections
from .lib.detection import detect_landmarks
from .lib.loader import load_ble_observations


def run_landmark_correction(
    trajectory: list[list[float]],
    t_at_steps: list[float],
    settings: BleLandmarkSettings,
) -> LandmarkCorrectionResult | None:
    """BLE ランドマーク補正を実行する。無効なら ``None`` を返す。"""
    if not settings.enabled:
        return None
    observations = load_ble_observations(settings.data_path)
    detections = detect_landmarks(observations, settings)
    return apply_landmark_corrections(
        trajectory,
        t_at_steps,
        detections,
        settings,
        str(settings.data_path),
    )
