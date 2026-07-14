"""Matplotlib の実行環境依存設定。

役割:
    Matplotlib がフォント等のキャッシュを書き込めるよう、未設定時の
    ``MPLCONFIGDIR`` を OS の一時ディレクトリ配下へ設定する。
依存元:
    標準ライブラリの ``os``、``pathlib``、``tempfile`` だけを使用する。
利用先:
    パッケージの CLI エントリポイントが Matplotlib を使う解析モジュールより先に
    ``configure_matplotlib_cache`` を呼び出す。
処理フロー:
    環境変数が既にあれば維持し、なければ一時キャッシュディレクトリを作成して設定する。
"""

import os
from pathlib import Path
from tempfile import gettempdir


def configure_matplotlib_cache() -> None:
    """Matplotlib のキャッシュ先を必ず書き込み可能な場所にする。"""
    if "MPLCONFIGDIR" in os.environ:
        return

    cache_dir = Path(gettempdir()) / "rikka-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
