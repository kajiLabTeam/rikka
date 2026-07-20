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

5. 各 seed の `wall_crossings` と `recovery_failures` がともに 0で、正解軌跡がある場合は `terminal_direction_failure` も false であることを合格条件にする。
6. `arc_rmse_m`、`endpoint_error_m`、`terminal_direction_error_deg`、`terminal_opposed_fraction`、`estimated_length_m`、`max_position_spread_m`、`checkpoint_replays` を seed 別に示し、数値項目は中央値と最大値も集計する。RMSEや終点誤差には、ユーザー指定または既存ベースラインがない限り固定の合格閾値を設けない。
7. 視覚比較が必要な場合だけ `--plot-path output/diagnostics/<name>.png` を追加する。確認後、この評価で生成した画像と新規出力だけを削除する。
8. 実行コマンド、入力、seed、合否、集計値、seed間のばらつき、削除した生成物を報告する。

## 手法横断ベンチマーク

通常PDRとPFの方式を複数データ・複数seedで同時比較する場合は、次を使う。

```sh
MPLBACKEND=Agg uv run python agent/agent_benchmark_pdr_pf_methods.py
```

- 必要入力は、既定の5つの反復計測、共通正解軌跡、フロアマップである。
- 予測尤度temperingを比較するときは
  `--pf-methods adaptive-causal --motion-predictive-weight-powers 0 0.1` を指定する。
- `--output /tmp/<name>.json` またはCSVを指定した場合は、集計確認後にその一時ファイルを削除する。
- PFは全seedの `wall_crossings == 0`、`recovery_failures == 0`、正解軌跡がある場合は `terminal_direction_failure == false` を必須条件とし、
  全体中央値だけでなく方式別最大RMSEとデータ内seed標準偏差も比較する。
- 一部データだけの改善で既定方式を変更せず、各データの最大値非悪化を確認する。

候補軌跡とmanifestを保存し、反転診断と複数計測の代表軌跡を作る場合は次を使う。

```sh
MPLBACKEND=Agg uv run python agent/agent_benchmark_pdr_pf_methods.py \
  --pdr-methods adaptive-causal robust-causal \
  --pf-methods adaptive-causal \
  --motion-predictive-weight-powers 0.1 \
  --pf-path-selections current sequence \
  --candidate-dir /tmp/rikka_candidates \
  --output /tmp/rikka_benchmark.json
uv run python agent/agent_diagnose_heading_reversal.py \
  --manifest /tmp/rikka_candidates/candidates.csv \
  --truth-csv 'input/correct_path/1turn_rightsidestep_3turn_leftsidestep/walk_trace (3).csv' \
  --output-dir /tmp/rikka_reversal --require-standard-coverage
MPLBACKEND=Agg uv run python agent/agent_build_consensus_trajectory.py \
  --manifest /tmp/rikka_candidates/candidates.csv \
  --diagnostics-json /tmp/rikka_reversal/reversal_diagnostics.json \
  --output-dir output/diagnostics/<name>
```

- 実反転は角度wrapと分け、135度以上の逆行が2点以上続く区間として確認する。加えて終端15%の方向差、基準方向へのprojection cosine、逆向き接線割合を確認する。
- medoidには壁交差、recovery failure、実反転がある候補を含めない。3計測以上では正解を使わない計測・方式等重みの終端方向コンセンサスから90度を超えて外れる候補も除外する。
- 一時候補と診断は確認後に `/tmp` から削除し、ユーザー向け代表軌跡だけを残す。

## 失敗時の扱い

- 必要な既定入力がなければ代替データを推測で選ばず、不足しているパスを報告する。
- 壁交差、recovery failure、終端方向failureが1件でもあれば失敗 seedとして扱い、平均値で隠さない。
- 一部 seed だけの改善を全体改善と断定しない。中央値、最大値、失敗 seed を併記する。
- `uv` が既定キャッシュの権限エラーで失敗した場合だけ、同じコマンドを `UV_CACHE_DIR=.uv-cache` 付きで再実行する。
- 検証前から存在した `output/` と `input/` は変更しない。
