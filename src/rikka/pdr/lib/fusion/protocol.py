"""通常 PDR の推定方式レジストリ。

役割:
    legacy / adaptive / robust 推定を同じ選択表から取得できるようにする。
依存元:
    同じ ``fusion`` 領域の各実装を参照する。
利用先:
    PDR pipeline が設定された推定方式を分岐なしで選択するために使用する。
処理フロー:
    設定名をレジストリで解決し、選択された推定 callable を呼び出し元へ返す。
"""

from collections.abc import Callable
from typing import Protocol

from .adaptive import estimate_adaptive_pdr
from .legacy import estimate_legacy_pdr
from .robust import resolve_step_directions


class MotionEstimator(Protocol):
    """運動状態推定実装が満たす callable 契約。"""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """入力された歩観測を推定結果へ変換する。"""
        ...


MOTION_ESTIMATORS: dict[str, Callable[..., object]] = {
    "legacy": estimate_legacy_pdr,
    "adaptive": estimate_adaptive_pdr,
    "robust": resolve_step_directions,
}

