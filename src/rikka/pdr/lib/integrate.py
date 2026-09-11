"""ステップ列積分のPDR互換ファサード。

役割:
    共通領域へ移した ``integrate_steps`` を従来の import path でも公開する。
依存元:
    ``common.lib.integrate`` の共有実装を使用する。
利用先:
    PDR pipeline の既存呼び出し元から互換目的で使用される。
処理フロー:
    呼び出しを共有実装へそのまま転送する。
"""

from ...common.lib.integrate import integrate_steps

__all__ = ["integrate_steps"]
