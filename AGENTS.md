# AGENT.md

このリポジトリで作業する AI エージェント向けのガイドです。

## 基本方針


- 回答は日本語で行ってください。
- コードコメントを書く場合、説明文は日本語で書いてください。
- 既存の未コミット変更はユーザーの作業として扱い、明示的な依頼なしに戻さないでください。
- `input/` には実験データ、`output/` には実行結果が入ります。不要な大容量ファイルや生成物をコミットしないでください。

## プロジェクト概要

`rikka` は、スマートフォンの加速度計・ジャイロスコープ CSV から歩行軌跡を推定する PDR（Pedestrian Dead Reckoning）ライブラリです。

- パッケージ管理・実行には `uv` を使います。
- Python 要件は `>=3.14` です。
- CLI エントリポイントは `rikka = "rikka:main"` です。
- 入力データは phyphox 形式の `Accelerometer.csv` と `Gyroscope.csv` を想定します。
- 通常 PDR と、パーティクルフィルタ付きマップマッチングの2系統があります。

## 主要構成

- `src/rikka/__init__.py`: Click ベースの CLI 定義。`run`、`pdr`、`particle`、`sensor` コマンドを提供します。
- `src/rikka/config.py`: 入力データ、フロアマップ、歩幅推定、パーティクルフィルタの既定値を定義します。
- `src/rikka/analyze/pdr.py`: センサーデータ読み込み、前処理、ステップ検出、歩幅推定、軌跡推定、CSV/画像出力の中心処理です。
- `src/rikka/analyze/particle_filter.py`: パーティクルフィルタとフロアマップ上のマップマッチング、アニメーション出力を扱います。
- `src/rikka/analyze/sensor_plot.py`: センサー波形と歩幅グラフの可視化を担当します。
- `src/rikka/ping.py`: 接続確認用の `ping()` を提供します。
- `input/`: サンプル・実験用センサーデータとフロアマップ画像を置く場所です。
- `output/`: `rikka run` / `rikka particle` の実行結果がタイムスタンプ付きで出力されます。

## 入力と出力

入力データは次の形で配置します。

```text
input/
└── my_walk/
    ├── Accelerometer.csv
    └── Gyroscope.csv
```

`Accelerometer.csv` は `Time (s)`, `Acceleration x (m/s^2)`, `Acceleration y (m/s^2)`, `Acceleration z (m/s^2)` を想定します。
`Gyroscope.csv` は `Time (s)`, `Gyroscope x (rad/s)`, `Gyroscope y (rad/s)`, `Gyroscope z (rad/s)` を想定します。
`X (m/s^2)` / `X (rad/s)` 形式の列名にも対応しています。

通常 PDR とパーティクルフィルタは `output/<timestamp>/` に `trajectory.csv`、`step_lengths.csv`、グラフ画像を保存します。
`sensor` コマンドは入力フォルダ内に `sensor_plot.png` を保存します。

## 開発コマンド

依存関係を同期します。

```sh
uv sync --all-groups
```

通常 PDR を実行します。

```sh
uv run rikka run
```

`run` と同じ処理を別名で実行します。

```sh
uv run rikka pdr
```

パーティクルフィルタ付きで実行します。MP4 出力には ffmpeg が必要です。

```sh
uv run rikka particle
```

センサーデータを可視化します。

```sh
uv run rikka sensor
```

フロアマップや起点を CLI から指定できます。

```sh
uv run rikka run -d input/my_walk -f input/map.png --origin-px 1000 500 --scale 0.01 --direction 90 --no-plot
```

## 品質チェック

AI エージェントが `uv` コマンドを実行するときは、ホームディレクトリ配下の
キャッシュ権限で止まらないよう、リポジトリ内キャッシュを使ってください。

```sh
UV_CACHE_DIR=.uv-cache uv run pytest
```

同様に `uv run ruff ...`、`uv run mypy ...`、`uv build` なども
`UV_CACHE_DIR=.uv-cache` を付けて実行してください。ユーザーはこの方針を承認済みです。

フォーマットします。

```sh
uv run ruff format
```

リントします。

```sh
uv run ruff check
```

自動修正付きでリントします。

```sh
uv run ruff check --fix
```

型チェックします。

```sh
uv run mypy src/
```

テストを実行します。

```sh
uv run pytest
```

パッケージをビルドします。

```sh
uv build
```

## コミット前と CI

リポジトリ管理の Git hook を使う場合は、最初に次を設定します。

```sh
git config core.hooksPath .githooks
```

CI と同等の pre-commit チェックを手元で実行します。

```sh
uv run pre-commit run --all-files
```

CI は GitHub Actions で `uv sync --all-groups`、`pre-commit run --all-files`、`uv build` を実行します。

## 実装時の注意

- `pyproject.toml` の Ruff 設定は行長 88、ダブルクォート、スペースインデントです。
- Mypy は strict 設定ですが、`disallow_untyped_defs = false` です。既存コードの型付け方針に合わせてください。
- `config.py` の既定値は CLI のデフォルトにも使われます。設定変更は CLI 挙動にも影響します。
- `pdr.run()` は `df_acc` と `df_gyro` を両方渡すか、両方省略する必要があります。片方だけ渡すと `ValueError` になります。
- プロット処理は `plt.show()` を呼びます。CI やバッチ確認では `--no-plot` または `plot=False` を使ってください。
- `particle` は ffmpeg がない場合、GIF 出力へフォールバックします。
