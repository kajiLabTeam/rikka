"""解析結果のコンソール表示。

役割:
    pipeline 内へ散在していた進捗・結果表示を出力領域へ集約する。
依存元:
    ``TrajectoryResult`` の確定済み結果だけを受け取る。
利用先:
    ``plot.pipeline`` がCSV・図の書き出し前に概要を表示する。
処理フロー:
    歩数と各軌跡点を既存と同じ書式で標準出力へ表示する。
"""

from ...common.lib.models import TrajectoryResult


def print_trajectory_summary(result: TrajectoryResult) -> None:
    """歩数と各軌跡点を表示する。"""
    print(f"Peaks detected: {len(result.prepared.step_detection.peaks)}")
    print(f"Steps used: {len(result.step_lengths)}")
    for index, (x, y) in enumerate(result.trajectory):
        print(f"step {index}: ({x:.3f}, {y:.3f})")
