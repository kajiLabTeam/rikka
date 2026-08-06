# Rikka コードリード資料

## 目的

この資料は、画面共有しながら現在の実装を追い、センサー CSV から通常 PDR・
particle filter・成果物までのデータ受け渡しと採用中の計算を確認するための入口です。
一般的な PDR の解説ではなく、実際の関数、値、分岐を中心に記載します。

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

## システム全体を8行で説明

1. Click CLI が phyphox 形式の加速度・ジャイロ CSV を読みます。
2. 列名を `t,x,y,z` に統一し、重力・線形・上下・水平加速度を作ります。
3. ジャイロ bias を補正して角速度を実時刻積分し、歩行イベントを検出します。
4. 各歩で body heading、水平加速度由来 motion heading、歩幅候補を作ります。
5. 標準の `adaptive + causal` が運動状態・方位・歩幅の事後分布を更新します。
6. 通常 PDR は方位と歩幅を原点から積算します。
7. `particle` は同じ `PreparedPdrSteps` を粒子へ入力し、地図制約と復旧を加えます。
8. CLI が共通結果を `output/<timestamp>/` の CSV・図・動画へ渡します。

## 現在の標準的な処理ルート

`rikka run` は `peak → gyro_accel_motion → clustered → adaptive/causal →
Weinberg → integrate_steps()` です。`rikka particle` はその同じ歩列へ、500粒子、
全通過画素の壁判定、ESS 0.5、map-aware recovery、guard付き `sequence` 選択を追加します。

注意: 現在のコードの `DATA_DIR` は
`input/sensor_data/hiroto/hiroto_1turn_rightsidestep_3turn_leftsidestep` です。
ルート `README.md` に書かれた `natsuki/1turn_rightsidestep_3turn_leftsidestep5` とは
一致しません。

## 資料内で使う記号

精度の表は `agent/EXPERIMENT_LOG.md` の表記を引き継いでいます。「無印」は
`input/sensor_data/natsuki/1turn_rightsidestep_3turn_leftsidestep`（末尾に数字なし）、
「5〜9」は同じルートの反復計測 `...leftsidestep5` 〜 `...leftsidestep9` を指します。

## コードリード時に重点的に確認する箇所

- 境界型 `common.lib.models.PreparedPdrSteps` と `TrajectoryResult`
- `pdr.lib.preparation.prepare_pdr_steps_with_settings()` の処理順
- `pdr.lib.fusion.protocol.MOTION_ESTIMATORS` の方式切替
- `particle.lib.runner.run_particle_steps()` の1歩ループ
- `particle.lib.evaluate_map.resolve_map_constraints()` の復旧順
- `plot.pipeline.write_outputs()` / `render()` との接続
- 標準値は `common/config/__init__.py` と CLI の両方で一致するか

## 未確認事項

- `DATA_DIR` を hiroto データへ変更した意図と、そのデータに対応する正解軌跡
- 現行既定入力に対する最新の同一条件精度（ログの主評価は natsuki 反復計測）
- センサーを取得する端末アプリ側の設定。リポジトリは CSV 読み込み以降のみ実装
- 各しきい値の端末・装着者をまたぐ校正根拠

詳細は [確認質問](50_questions.md) に集約しています。
