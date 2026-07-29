"""ステップ列を2次元座標列へ積分する唯一の実装。

役割:
    確定済みの歩方位と歩幅から原点始まりの軌跡を構築する。
依存元:
    ``common.lib.models.StepHeading`` と NumPy の三角関数を使用する。
利用先:
    PDR pipeline の legacy / adaptive / robust 推定結果から呼び出す。
処理フロー:
    各歩を入力順に加算し、従来と同じ浮動小数点演算順で座標を追加する。
"""

import numpy as np

from ...common.lib.models import StepHeading


def integrate_steps(
    headings: list[StepHeading],
    lengths: list[float],
) -> list[list[float]]:
    """確定方位と歩幅を原点から順に積分する。"""
    if len(headings) != len(lengths):
        raise ValueError("headings と lengths の長さが一致していません。")
    trajectory = [[0.0, 0.0]]
    for heading, length in zip(headings, lengths, strict=True):
        if heading.selected_heading is None:
            raise ValueError("selected_heading が未確定です。")
        trajectory.append(
            [
                trajectory[-1][0] + length * float(np.cos(heading.selected_heading)),
                trajectory[-1][1] + length * float(np.sin(heading.selected_heading)),
            ]
        )
    return trajectory
