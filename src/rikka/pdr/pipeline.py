"""センサー入力を通常PDR結果へ変換する pipeline。

役割:
    入力データを取得し、検証済み設定で共有歩列を準備して BLE 補正後の結果へまとめる。
依存元:
    common のセンサー入力・設定・結果型と ``lib.preparation`` を利用する。
利用先:
    CLI が最初に呼び、必要なら同じ準備済み歩列を particle pipeline へ渡す。
処理フロー:
    入力ペア確認、既定CSV読み込み、共有歩列準備、BLE補正、結果型構築の順に処理する。
"""

import pandas as pd

from ..ble.pipeline import run_landmark_correction
from ..common.lib.models import TrajectoryResult
from ..common.lib.sensors import load_sensor_data
from ..common.settings import PdrSettings
from .lib.preparation import prepare_pdr_steps_with_settings


def run_pdr(
    settings: PdrSettings,
    df_acc: pd.DataFrame | None = None,
    df_gyro: pd.DataFrame | None = None,
) -> TrajectoryResult:
    """センサー入力を共有ステップと通常PDR軌跡へ変換する。"""
    if (df_acc is None) != (df_gyro is None):
        raise ValueError("df_acc と df_gyro は両方渡すか、両方省略してください。")
    if df_acc is None and df_gyro is None:
        df_acc, df_gyro = load_sensor_data()
    if df_acc is None or df_gyro is None:
        raise RuntimeError("内部エラー: センサーデータが取得できませんでした。")

    prepared = prepare_pdr_steps_with_settings(df_acc, df_gyro, settings)
    landmark = run_landmark_correction(
        prepared.trajectory,
        prepared.t_at_steps,
        settings.landmark,
    )
    return TrajectoryResult(
        trajectory=prepared.trajectory if landmark is None else landmark.trajectory,
        step_lengths=prepared.step_lengths,
        t_at_steps=prepared.t_at_steps,
        step_headings=prepared.step_headings,
        prepared=prepared,
        landmark=landmark,
    )
