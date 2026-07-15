---
name: evaluate-rikka-particle-filter
description: Rikka の particle filter を固定 seed の回帰テストと正解軌跡に対する複数 seed 評価で検証する。particle filter、マップ衝突、resampling、recovery、heading noise、stride scale、particle diagnostics を変更・調査するときに使用する。通常 PDR だけの変更やドキュメントだけの変更には使用しない。
---

# Rikka Particle Filter 評価

particle filter の決定論的回帰を確認し、複数 seed の軌跡精度と地図制約を同じ手順で評価する。
根拠のない精度閾値を設けず、地図違反と復旧失敗を明確な失敗として扱う。

## 手順

1. `git status --short` と対象差分を読み、particle filter のどの処理が変わったかを特定する。
2. 検証前に `output/` 直下の既存項目を記録する。
3. 固定 seed、遷移判定、resampling、recovery、軌跡復元に関する回帰テストを実行する。

   ```sh
   uv run pytest tests/test_pdr_regressions.py -k "particle or resampling or recovery or transition or reconstruct"
   ```

4. 既定のセンサーデータ、正解軌跡、フロアマップが存在することを確認し、既存の評価スクリプトを既定 seed `[0, 1, 2, 10, 42, 100]` で実行する。

   ```sh
   MPLBACKEND=Agg uv run python agent/agent_evaluate_pf_ground_truth.py
   ```

5. 各 seed の `wall_crossings` と `recovery_failures` がともに 0 であることを合格条件にする。
6. `arc_rmse_m`、`endpoint_error_m`、`estimated_length_m`、`max_position_spread_m`、`checkpoint_replays` を seed 別に示し、数値項目は中央値と最大値も集計する。RMSEや終点誤差には、ユーザー指定または既存ベースラインがない限り固定の合格閾値を設けない。
7. 視覚比較が必要な場合だけ `--plot-path output/diagnostics/<name>.png` を追加する。確認後、この評価で生成した画像と新規出力だけを削除する。
8. 実行コマンド、入力、seed、合否、集計値、seed間のばらつき、削除した生成物を報告する。

## 失敗時の扱い

- 必要な既定入力がなければ代替データを推測で選ばず、不足しているパスを報告する。
- 壁交差または recovery failure が1件でもあれば失敗 seed として扱い、平均値で隠さない。
- 一部 seed だけの改善を全体改善と断定しない。中央値、最大値、失敗 seed を併記する。
- `uv` が既定キャッシュの権限エラーで失敗した場合だけ、同じコマンドを `UV_CACHE_DIR=.uv-cache` 付きで再実行する。
- 検証前から存在した `output/` と `input/` は変更しない。
