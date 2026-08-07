# Rikka コードリード資料

## 目的

この資料は、画面共有しながら現在の実装を追うための入口です。
センサー CSV から通常 PDR・particle filter・成果物までのデータ受け渡しと、
採用中の計算を確認できます。一般的な PDR の解説ではなく、実際の関数、値、
分岐を中心に記載します。

## 推奨する読む順番

1. [システム全体フロー](01_system-flow.md)
2. [実行開始から終了](02_entry-and-execution.md)
3. [主要データ構造](03_data-structures.md)
4. [コードリード進行表](40_code-reading-guide.md)
5. 下表の処理別資料

| 区分 | 資料 |
|---|---|
| 入力・推定 | [入力と前処理](10_data-loading-and-preprocessing.md) / [歩検出](11_step-detection.md) / [方位推定](12_heading-estimation.md) / [運動状態推定](13_motion-state-estimation.md) / [歩幅と軌跡](14_step-length-and-trajectory.md) |
| 地図拘束 | [particle filter](15_particle-filter.md) / [マップ制約と復旧](16_map-matching-and-recovery.md) |
| 結果利用 | [評価](18_evaluation.md) / [保存・表示](19_output-and-visualization.md) |
| 横断整理 | [標準設定と切替](30_active-configurations.md) / [精度比較](31_accuracy-comparison.md) |
| 会議用 | [確認質問](50_questions.md) / [今後の改善案](60_future-improvements.md) |

## コードへのリンク

単独で記す関数名・型名・定数名は、資料ごとの初出をコードへリンクしています。
モジュールを含むドット表記と、実装追跡の対象となるファイルパスは、同じ資料内で
再登場してもリンクしています。
`src/` 配下への行番号付きリンクは GitHub で定義行を開けます。ローカルエディタでの
表示位置は利用環境に依存するため、コードリード前に次のコマンドで照合してください。

```sh
uv run python agent/agent_check_doc_links.py
```

定義行がずれたリンクは、「現在の定義行はL○○」という情報と併せて報告されます。

このリンク方針の例外として、`agent/` は `.gitignore` で除外されたローカル専用領域です。
`agent/EXPERIMENT_LOG.md`、`agent/agent_*.py`、照合スクリプト自体は手元の
クローンにのみ存在し、GitHub では開けません。これらを確認する画面共有は
ローカルの資料とコードで行ってください。

## システム全体を8ステップで説明

1. Click CLI が引数を解析し、[`cli.commands.run()`](../../src/rikka/cli/commands.py#L81) が設定を組み立てます。
2. [`pdr.pipeline.run_pdr()`](../../src/rikka/pdr/pipeline.py#L21) が phyphox 形式の
   加速度・ジャイロ CSV を読み、列名を `t,x,y,z` に統一します。
3. 前処理で重力・線形・上下・水平加速度を作り、bias 補正後の角速度を実時刻積分します。
4. 加速度の `low_lin_norm` のピークから歩行イベントを検出します。
5. 各歩で body heading、水平加速度由来 motion heading、歩幅候補を作ります。
6. 標準の `adaptive + causal` が運動状態・方位・歩幅の事後分布を更新します。
7. 通常 PDR は方位と歩幅を積算し、`rikka particle` は同じ
   [`PreparedPdrSteps`](../../src/rikka/common/lib/models.py#L230) に地図制約と復旧を加えます。
8. [`cli.commands.run()`](../../src/rikka/cli/commands.py#L81) が共通結果を
   [`plot.pipeline`](../../src/rikka/plot/pipeline.py) へ渡し、CSV・図・動画を
   `output/<timestamp>/` に出力します。

## 現在の標準的な処理ルート

`rikka run` の標準処理ルートは、`peak` → `gyro_accel_motion` → `clustered` →
`adaptive/causal` → `Weinberg` →
[`integrate_steps()`](../../src/rikka/pdr/lib/integrate.py#L18) の順です。
`rikka particle` は同じ歩列へ、500粒子、全通過画素の壁判定、ESS 0.5、
map-aware recovery、guard付き `sequence` 選択を追加します。

## 資料内で使う略称

精度の表では、[`agent/EXPERIMENT_LOG.md`](../../agent/EXPERIMENT_LOG.md) の表記に
合わせています。「無印」は
[`input/sensor_data/natsuki/1turn_rightsidestep_3turn_leftsidestep`](../../input/sensor_data/natsuki/1turn_rightsidestep_3turn_leftsidestep)（末尾に数字なし）、
「5〜9」は同じルートの反復計測 `...leftsidestep5` 〜 `...leftsidestep9` を指します。

## コードリード時に重点的に確認する箇所

- 境界型 [`common.lib.models.PreparedPdrSteps`](../../src/rikka/common/lib/models.py#L230) と [`TrajectoryResult`](../../src/rikka/common/lib/models.py#L280) の役割
- [`pdr.lib.preparation.prepare_pdr_steps_with_settings()`](../../src/rikka/pdr/lib/preparation.py#L111) の処理順
- [`pdr.lib.fusion.protocol.MOTION_ESTIMATORS`](../../src/rikka/pdr/lib/fusion/protocol.py#L117) の方式切替
- [`particle.lib.runner.run_particle_steps()`](../../src/rikka/particle/lib/runner.py#L74) の1歩ループ
- [`particle.lib.evaluate_map.resolve_map_constraints()`](../../src/rikka/particle/lib/evaluate_map.py#L98) の復旧順
- [`plot.pipeline.write_outputs()`](../../src/rikka/plot/pipeline.py#L51) /
  [`render()`](../../src/rikka/plot/pipeline.py#L128) への出力接続
- [`common/config/__init__.py`](../../src/rikka/common/config/__init__.py) と CLI の標準値の一致

## 未確認事項

- 現行既定入力に対する最新の同一条件精度（ログの主評価は natsuki 反復計測）
- センサー取得端末アプリ側の設定（リポジトリの実装範囲は CSV 読み込み以降）
- 各しきい値の端末・装着者をまたぐ校正根拠

詳細は [確認質問](50_questions.md) に集約しています。
