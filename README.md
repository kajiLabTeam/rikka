# rikka

# 開発コマンド

## 実行コマンド
```sh
$ uv run rikka
```


# コミット前に以下のコマンドをする

`git commit` 時に format を自動反映したい場合は、最初に repo の hook を有効化します。

```sh
git config core.hooksPath .githooks
```

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

`.githooks/pre-commit` を有効化している場合、commit 時は自動修正された内容を hook 側で再 stage してから再チェックするため、format 修正だけなら commit が止まりにくくなります。
