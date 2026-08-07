"""rikka の公開 API と CLI エントリポイント。

Click のオプション・コマンド定義は ``rikka.cli.options``、解析処理は
``rikka.cli.commands`` に置き、このモジュールは公開入口だけを提供する。
"""

from .cli.options import cli
from .matplotlib_config import configure_matplotlib_cache
from .ping import ping

__all__ = ["main", "ping"]


def main() -> None:
    """Click CLI を起動する。"""
    configure_matplotlib_cache()
    cli()
