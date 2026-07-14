"""パッケージ接続確認用の最小 API。

役割:
    rikka が正しく import できることを確認するため、固定メッセージを返す。
依存元:
    外部ライブラリや他のプロジェクトモジュールには依存しない。
利用先:
    パッケージ直下の ``rikka.__init__`` が ``ping`` を再公開し、利用者や smoke test が
    インストール・import 状態の確認に使用する。
処理フロー:
    ``ping`` が引数なしで呼ばれると ``"Hello, rikka"`` を返す。
"""


def ping() -> str:
    """接続確認用の関数。引数なしで "Hello, rikka" を返す。"""
    return "Hello, rikka"
