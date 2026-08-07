"""補正済みの歩列をそのまま採用する legacy 推定。

役割:
    legacy モードを adaptive / robust と同じレジストリから選択可能にする。
依存元:
    ``common`` の歩方位型だけを使用する。
利用先:
    ``fusion.protocol.MOTION_ESTIMATORS`` から legacy 選択時に呼ばれる。
処理フロー:
    入力された方位・歩幅を変更せず返す。
"""

from ....common.lib.models import StepHeading


def estimate_legacy_pdr(
    step_headings: list[StepHeading],
    step_lengths: list[float],
) -> tuple[list[StepHeading], list[float]]:
    """既存の確定結果をそのまま返す。"""
    return step_headings, step_lengths
