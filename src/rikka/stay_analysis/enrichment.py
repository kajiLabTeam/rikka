"""trajectoryの速度算出と滞在判定。

役割:
    標準trajectoryへ到達速度と行単位の滞在フラグを追加する。
依存元:
    validationから入力schemaとparameter検証を取得する。
利用先:
    Nozomiが派生CSVを作り、grid処理へ渡すために使用する。
処理フロー:
    入力を検証し、3行目以降の到達速度を計算して閾値以下を滞在と判定する。
"""

import numpy as np
import pandas as pd

from rikka.stay_analysis.validation import validate_parameter, validate_trajectory


def enrich_trajectory(
    dataframe: pd.DataFrame,
    *,
    speed_threshold_mps: float = 0.5,
) -> pd.DataFrame:
    """trajectoryを検証し、到達速度と行単位の滞在判定を追加する。"""
    threshold = validate_parameter(
        speed_threshold_mps,
        name="speed_threshold_mps",
        minimum=0,
        maximum=2,
    )
    validate_trajectory(dataframe)

    original = dataframe.drop(columns=["speed_mps", "is_stay"], errors="ignore")
    enriched = original.copy()
    speed = np.full(len(enriched), np.nan, dtype=float)
    if len(enriched) >= 3:
        timestamps = enriched["rikka_timestamp_s"].to_numpy(dtype=float)
        x = enriched["rikka_x"].to_numpy(dtype=float)
        y = enriched["rikka_y"].to_numpy(dtype=float)
        speed[2:] = np.hypot(np.diff(x)[1:], np.diff(y)[1:]) / np.diff(timestamps[1:])

    elapsed = enriched["rikka_timestamp_s"].to_numpy(dtype=float)
    enriched["speed_mps"] = speed
    enriched["is_stay"] = np.isfinite(speed) & (elapsed > 0.5) & (speed <= threshold)
    return enriched
