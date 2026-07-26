"""Matplotlib の実行環境依存設定。

役割:
    Matplotlib がフォント等のキャッシュを書き込めるよう、未設定時の
    ``MPLCONFIGDIR`` を OS の一時ディレクトリ配下へ設定し、日本語を描画できる
    フォントを実行環境から選択する。
依存元:
    標準ライブラリの ``os``、``pathlib``、``tempfile`` を使用する。
    日本語フォント設定時だけ Matplotlib を遅延 import する。
利用先:
    パッケージの CLI エントリポイントが Matplotlib を使う解析モジュールより先に
    ``configure_matplotlib_cache`` を呼び出す。
処理フロー:
    環境変数が既にあれば維持し、なければ一時キャッシュディレクトリを作成して設定する。
    描画開始時には利用可能な日本語フォント候補を順に調べて適用する。
"""

import os
from pathlib import Path
from tempfile import gettempdir

_JAPANESE_FONT_CANDIDATES = (
    "Hiragino Sans",
    "Hiragino Maru Gothic Pro",
    "Yu Gothic",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "TakaoGothic",
)
_JAPANESE_FONT_CONFIGURED = False


def configure_matplotlib_cache() -> None:
    """Matplotlib のキャッシュ先を必ず書き込み可能な場所にする。"""
    if "MPLCONFIGDIR" in os.environ:
        return

    cache_dir = Path(gettempdir()) / "rikka-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def configure_japanese_font() -> None:
    """Matplotlib で利用可能な日本語フォントを1回だけ設定する。"""
    global _JAPANESE_FONT_CONFIGURED  # noqa: PLW0603
    if _JAPANESE_FONT_CONFIGURED:
        return

    from matplotlib import font_manager, pyplot  # noqa: PLC0415

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in _JAPANESE_FONT_CANDIDATES:
        if font_name in available_fonts:
            pyplot.rcParams["font.family"] = [font_name]
            pyplot.rcParams["axes.unicode_minus"] = False
            break

    _JAPANESE_FONT_CONFIGURED = True
