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
