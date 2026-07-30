"""Click CLI の実行処理。

役割:
    コマンドライン引数の定義と、領域別 pipeline を呼び出す実行処理を提供する。
依存元:
    ``commands`` から PDR / particle filter の互換実行関数を取得する。
利用先:
    パッケージ直下の CLI エントリポイントと後方互換 API から使用される。
処理フロー:
    CLI が解釈した設定を ``commands`` へ渡し、解析結果と成果物を生成する。
"""

from .. import _click_cli
from .commands import run

# ``from rikka import cli`` の既存コードがこのサブパッケージを取得しても、
# Click の Command として扱えるよう最小限の属性を転送する。
name = _click_cli.name
main = _click_cli.main

__all__ = ["main", "name", "run"]
