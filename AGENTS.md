# AGENTS.md

このリポジトリで作業する AI エージェント向けのガイドです。

## 基本方針

- 回答は日本語で行ってください。
- コードコメントを書く場合、説明文は日本語で書いてください。
- 既存の未コミット変更はユーザーの作業として扱い、明示的な依頼なしに戻さないでください。
- `input/` には実験データ、`output/` には実行結果が入ります。不要な大容量ファイルや生成物をコミットしないでください。
- AI エージェントが検証のために生成した `output/` 配下の実行結果だけを、検証後に削除してください。
- 読み取り専用の調査と既存テストは、合理的な仮定を置いて進めてください。
- 使用する実験データ、座標系、評価基準が結果を左右する場合は、実装や長時間評価の前に確認してください。
- 依存関係の追加、設定の既定値変更、実験データの削除は、実行前に確認してください。

## プロジェクト概要

`rikka` は、スマートフォンの加速度計・ジャイロスコープ CSV から歩行軌跡を推定する PDR（Pedestrian Dead Reckoning）ライブラリです。

- パッケージ管理・実行には `uv` を使います。
- Python 要件は `>=3.14` です。
- CLI エントリポイントは `rikka = "rikka:main"` です。
- 入力データは phyphox 形式の `Accelerometer.csv` と `Gyroscope.csv` を想定します。
- 通常 PDR と、パーティクルフィルタ付きマップマッチングの2系統があります。

## 構成と設計契約

詳しい入力形式、データフロー、設定、出力は `README.md` を参照してください。

- `src/rikka/__init__.py` は Click ベースの CLI を定義します。
- `src/rikka/config.py` の既定値は CLI のデフォルトにも使われます。
- `src/rikka/analyze/pdr/` は通常 PDR の実装で、`rikka.analyze.pdr` は互換 facade です。
- `src/rikka/analyze/particle_filter.py` はマップマッチングと粒子フィルタを扱います。
- 通常 PDR と particle filter は `prepare_pdr_steps()` のステップ情報を共有します。
- particle filter から PDR 内部処理を利用する場合は `pdr/particle_api.py` を bridge にします。
- `scripts/agent_*.py` はエージェントの診断・検証用で、通常のライブラリ API ではありません。

## 開発・品質チェック

パッケージ管理とコマンド実行には `uv` を使用します。

```sh
uv sync --all-groups
uv run ruff format
uv run ruff check
uv run mypy src/
uv run pytest
uv build
```

`uv` の既定キャッシュ先に対する権限エラーが発生した場合に限り、同じコマンドへ
`UV_CACHE_DIR=.uv-cache` を付けて再実行してください。

CI と同等のチェックは次のとおりです。CI、依存関係、ビルド設定へ影響する変更では必ず実行してください。

```sh
uv run pre-commit run --all-files
uv build
```

Git hook を利用する場合は `git config core.hooksPath .githooks` を設定します。

## 変更別の最低検証

- Python 実装: 関連テストと `uv run ruff check`
- heading、sidestep、step detection、step length、trajectory: `tests/test_pdr_regressions.py` と必要に応じた代表データ実行
- particle filter: 固定 seed の回帰テストと複数 seed 評価
- CLI、`config.py` の既定値: `tests/test_smoke.py` と該当コマンドの `--help`
- CI、依存関係、ビルド設定: `uv run pre-commit run --all-files` と `uv build`
- ドキュメントのみ: 原則テスト不要。ただし記載コマンドと現在の実装を照合する

CI やバッチ確認では `--no-plot` または `plot=False` を使ってください。

## 生成物の扱い

- 検証前に既存の `output/` を確認してください。
- 自分が生成した出力ディレクトリだけを記録し、検証後に削除してください。
- 検証前から存在した `output/` や `input/` 内のデータを削除しないでください。

## 実装時の注意

- `src/` 配下に新しい Python ファイルを作成する場合は、ファイル先頭に日本語の
  モジュール docstring を置いてください。docstring には少なくとも「役割」、
  「依存元（どのモジュールから何を取得するか）」、「利用先（どこから、何のために
  使用されるか）」、「処理フロー」を記載してください。既存のモジュール docstring が
  あるファイルでは、別の説明コメントを追加せず、その docstring に内容を統合してください。
- `pyproject.toml` の Ruff 設定は行長 88、ダブルクォート、スペースインデントです。
- Mypy は strict 設定ですが、`disallow_untyped_defs = false` です。既存コードの型付け方針に合わせてください。
- `config.py` の既定値は CLI のデフォルトにも使われます。設定変更は CLI 挙動にも影響します。
- `pdr.run()` は `pdr/pipeline.py` が実体です。`pdr/__init__.py` は互換 facade なので、外部互換を壊さないよう既存 import を維持してください。
- `pdr.run()` は `df_acc` と `df_gyro` を両方渡すか、両方省略する必要があります。片方だけ渡すと `ValueError` になります。
- 通常 PDR と particle filter で共有するステップ情報は `prepare_pdr_steps()` が作ります。particle 側で同じ heading / step length 推定を重複実装しないでください。
- `particle_filter.py` から PDR 側の内部処理を使う場合は、`pdr/particle_api.py` に bridge を追加してから利用してください。`pdr/__init__.py` の private re-export へ直接依存しないでください。
- `scripts/agent_*.py` はエージェント検証用です。恒常的な機能として扱わず、必要な検証目的・入力データ・出力先が分かる名前と docstring を保ってください。
- プロット処理は `plt.show()` を呼びます。CI やバッチ確認では `--no-plot` または `plot=False` を使ってください。
- `particle` は ffmpeg がない場合、GIF 出力へフォールバックします。
