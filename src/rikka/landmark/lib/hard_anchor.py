"""確定位置イベントの読み込み。

役割: 検出方式に依存しない根拠付き位置観測をJSONから構築する。
依存元: commonのHardPositionObservationと標準JSON処理を使用する。
利用先: particle CLIが確定位置ファイルを実行APIへ渡すために使用する。
処理フロー: JSON配列を読み、各観測の必須項目と型を検証して不変型へ変換する。
"""

import json
from pathlib import Path

from ...common.lib.models import HardPositionObservation


def load_hard_positions(path: str | Path) -> tuple[HardPositionObservation, ...]:
    """配列形式のJSONを読み、座標の丸めや時刻の自動割当を行わず返す。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("確定位置JSONはイベントの配列を指定してください")
    result = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("確定位置イベントはオブジェクトを指定してください")
        try:
            observation = HardPositionObservation(**item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"確定位置イベントの形式が不正です: {exc}") from exc
        result.append(observation)
    return tuple(result)
