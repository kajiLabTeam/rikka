"""解析結果のコンソール表示。

役割:
    pipeline 内へ散在していた進捗・結果表示と BLE 補正サマリを出力領域へ集約する。
依存元:
    ``TrajectoryResult`` の確定済み結果だけを受け取る。
利用先:
    ``plot.pipeline`` がCSV・図の書き出し前に概要を表示する。
処理フロー:
    歩数と各軌跡点を既存と同じ書式で表示し、BLE 補正時だけ診断内容を追記する。
"""

from ...common.lib.models import LandmarkCorrectionResult, TrajectoryResult


def print_landmark_summary(landmark: LandmarkCorrectionResult | None) -> None:
    """BLE ランドマークの検出件数と補正内容を表示する。"""
    if landmark is None:
        return
    applied = [item for item in landmark.corrections if item.applied]
    print(
        "Landmark correction: "
        f"threshold={landmark.rssi_threshold_dbm:.1f} dBm "
        f"detections={landmark.detection_count} "
        f"applied={len(applied)} "
        f"discarded={landmark.discarded_count} "
        f"data={landmark.data_path}"
    )
    for correction in landmark.corrections:
        mark = "*" if correction.applied else " "
        print(
            f"  {mark} step {correction.step_index} "
            f"t={correction.timestamp_s:.2f}s "
            f"{correction.beacon_id} rssi={correction.rssi_dbm:.1f} dBm "
            f"({correction.before_x:.3f}, {correction.before_y:.3f}) -> "
            f"({correction.after_x:.3f}, {correction.after_y:.3f})"
        )


def print_trajectory_summary(result: TrajectoryResult) -> None:
    """歩数と各軌跡点を表示する。"""
    print(f"Peaks detected: {len(result.prepared.step_detection.peaks)}")
    print(f"Steps used: {len(result.step_lengths)}")
    for index, (x, y) in enumerate(result.trajectory):
        print(f"step {index}: ({x:.3f}, {y:.3f})")
    print_landmark_summary(result.landmark)
