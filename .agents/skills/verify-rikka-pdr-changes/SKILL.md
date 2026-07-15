---
name: verify-rikka-pdr-changes
description: Rikka の通常 PDR に関する変更を、関連テスト、全回帰テスト、代表センサーデータ、生成CSVで検証する。heading、sidestep、step detection、step length、gyro bias、trajectory、PDRの設定やCLI挙動を変更・調査するときに使用する。ドキュメントだけの変更や particle filter だけの変更には使用しない。
---

# Rikka PDR 変更検証

通常 PDR の変更範囲に合う検証を選び、単体回帰と実データ上の挙動を確認する。
既存の実験データと検証前から存在する生成物を保護する。

## 手順

1. `git status --short` と対象差分を読み、変更したモジュールと公開挙動を特定する。
2. 検証前に `output/` 直下の既存項目を記録する。削除対象はこの検証で新しく生成した項目だけに限定する。
3. [テスト対応表](references/test-map.md)を読み、変更領域に対応する絞り込みテストを実行する。
4. 絞り込みテストが通ったら、次を実行する。

   ```sh
   uv run pytest tests/test_pdr_regressions.py tests/test_smoke.py
   uv run ruff check
   ```

5. 数値計算、方位、横歩き、歩幅、既定値、出力形式が変わる場合は、ユーザー指定データを優先し、指定がなければ現在の既定データで実行する。

   ```sh
   MPLBACKEND=Agg uv run rikka run --no-plot
   ```

6. heading または sidestep の比較が必要で、スクリプト内の対象データが今回の変更に適合するときだけ、次を実行する。

   ```sh
   MPLBACKEND=Agg uv run python agent/agent_verify_heading_fix_comparison.py
   ```

7. 新しい出力の `trajectory.csv`、`step_lengths.csv`、`step_headings.csv` を変更内容に応じて確認する。行数、非有限値、方位の不連続、movement type、終点を確認し、期待値を推測で合格扱いしない。
8. 記録した新規出力だけを正確なパス指定で削除する。検証前から存在した `output/` と `input/` は変更しない。
9. 実行コマンド、使用データ、成功・失敗、主要な観測値、削除した生成物を報告する。

## 判断規則

- `config.py` の変更では、CLI既定値と `--help` の回帰も確認する。
- `pdr/__init__.py` または facade の変更では、既存importと公開シンボルの互換性を確認する。
- particle filter と共有する情報は `prepare_pdr_steps()` を通り、重複実装されていないことを確認する。
- particle filter から PDR 内部処理への依存は `pdr/particle_api.py` を通ることを確認する。
- 実データ、座標系、正解基準が結果を左右し、ユーザー指定も既定値も使えない場合は、評価を推測せず確認を求める。
- `uv` が既定キャッシュの権限エラーで失敗した場合だけ、同じコマンドを `UV_CACHE_DIR=.uv-cache` 付きで再実行する。
