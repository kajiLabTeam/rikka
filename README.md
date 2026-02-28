# rikka

# 開発コマンド

## 実行コマンド
```sh
$ uv run rikka
```


# コミット前に以下のコマンドをする
## フォーマッター

```sh
$ uv run ruff format
```

## リンター

```sh
uv run ruff check
```

## リンター（自動修正）

```sh
uv run ruff check --fix
```

## 型チェック

```sh
uv run mypy src/
```
# CIで落ちたら

`uv run pre-commit run --all-files`

上記のコマンドを記述するとファイルの自動修正やエラー箇所の出力が行われます。出力されたエラー箇所を修正することでCIが通るようになると思います。

多くの場合は自動修正された変更をpushすることでCIに通るようになります。
