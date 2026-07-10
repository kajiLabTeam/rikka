"""Matplotlib の実行環境依存設定。"""

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
