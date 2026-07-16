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
- `agent/agent_*.py` はエージェントの診断・検証用で、通常のライブラリ API ではありません。
- 必要な診断・比較・回帰評価コードは `agent/` に追加し、繰り返し利用する場合は `.agents/skills/` 配下の Skill から呼び出して構いません。
- `agent/EXPERIMENT_LOG.md` は過去の試行、失敗、採用判断を残す実験ログです。

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

## サブエージェントの運用

- 20分以上かかる見込みで、独立した調査、実装、検証に分割できる場合は、サブエージェントによる並行化を検討してください。
- サブエージェントを開始する前に、担当範囲、編集可能なファイル、変更禁止範囲、期待する成果物を明示してください。
- 同じファイルを複数のサブエージェントへ同時に割り当てないでください。共有ワークツリー上の既存変更は、担当外のエージェントが編集・復元しないでください。
- 前工程の結果が必要な作業、判断基準が未確定な作業、短時間で完了する小さな修正は、無理に分割しないでください。
- 調査だけを担当するサブエージェントには、原則としてファイルを変更させず、根拠となるファイル、行、コマンド結果を報告させてください。
- メインエージェントは各結果と差分を確認してから統合し、重複実装、設計契約違反、未コミット変更との競合がないことを確認してください。
- 統合後の全体テスト、生成物の整理、`agent/EXPERIMENT_LOG.md` の更新、最終報告はメインエージェントが責任を持って行ってください。

## 生成物の扱い

- 検証前に既存の `output/` を確認してください。
- 自分が生成した出力ディレクトリだけを記録し、検証後に削除してください。
- 検証前から存在した `output/` や `input/` 内のデータを削除しないでください。

## 試行錯誤の記録

- PDR、particle filter、センサー処理の調査やパラメータ比較を始める前に、`agent/EXPERIMENT_LOG.md` を読み、関連語と過去エントリを検索してください。
- 新しい仮説、実データ評価、パラメータ比較で知見を得た場合は、成功・失敗にかかわらず同ファイルへ追記してください。
- 入力データ、比較条件、主要パラメータ、seed、実行コマンド、指標、結論、採否、再検証条件を記録してください。
- 単なる既存テストの再実行や新しい知見のない確認は追記せず、ログを作業日報にしないでください。
- 未コミット差分から結果を推測せず、評価が完了するまでは採用済みと記録しないでください。

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
- `agent/agent_*.py` はエージェント検証用です。恒常的な機能として扱わず、必要な検証目的・入力データ・出力先が分かる名前と docstring を保ってください。
- 新しい検証コードは `agent/agent_<目的>.py` の形式で命名し、モジュール docstring に役割、入力、出力、処理フローを記載してください。
- 検証コードを Skill から使う場合は、対象 Skill の `SKILL.md` に起動条件、実行コマンド、必要入力、合否判定、生成物の削除方法を記載してください。
- 検証コードは実行して動作確認し、通常機能に必要な処理を `agent/` だけへ実装しないでください。製品コードで共有すべき処理は `src/rikka/` に置いてください。
- プロット処理は `plt.show()` を呼びます。CI やバッチ確認では `--no-plot` または `plot=False` を使ってください。
- `particle` は ffmpeg がない場合、GIF 出力へフォールバックします。
