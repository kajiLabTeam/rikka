# `190/particle-filter-sidestep` ブランチ変更まとめ

## 2026-07-21追加: data9を含む6計測での汎化調整

- 歩行前の定速端末回転をgyro biasと誤認しないよう、`prewalk_guarded`を追加して標準化した。推定値が±0.003 rad/s以内だけ補正へ使い、それを超える場合は0へ戻す。
- PF再標本化後の方位ノイズを、各歩ではなく記録全体の移動観測信頼度から決める。通常は上限0.009 rad、高信頼記録は0.004 radまで下げる。
- 共通正解軌跡の対象へデータ9を追加し、6計測×6 seedで検証した。
- 従来→最終のRMSE中央値/最大値は、PDRが`5.175/8.149`→`3.313/5.183` m、PFが`4.907/13.291`→`2.064/6.554` m。最終PFは壁交差・recovery failure・終端方向failureがすべて0件だった。
- 粒子数増加、長期heading保持、固定rejuvenation noise、歩幅事前中心の増加は、一部seedまたは一部データを悪化させたため採用していない。

## 1. このブランチで達成したこと

このブランチでは、スマートフォンの加速度・ジャイロデータから歩行軌跡を作る処理を、
単純な「方位と固定歩幅の積み上げ」から、次の情報を共有して推定する構成へ拡張した。

- 通常歩行、左右の横歩き、旋回の運動状態
- ジャイロ由来の体・端末方向と、加速度由来の移動方向
- 歩ごとの歩幅と、その不確かさ
- フロアマップ上の壁・通路制約
- particle filter の粒子履歴、復旧履歴、運動状態尤度
- 複数計測・複数seed間の精度と終端方向の整合性

最終的な標準設定は、data9を含む6つの反復計測と6つの乱数seedで比較した結果から、
`prewalk_guarded + adaptive + causal + predictive weight 0.1 + guard付きsequence`
と、記録品質に応じたPF方位ノイズを組み合わせた構成とした。

## 2. 差分の範囲

作成時点の比較条件は次のとおり。

| 項目 | 値 |
|---|---|
| 比較元 | `origin/main` / `233e3e4` |
| 対象ブランチ | `190/particle-filter-sidestep` |
| 対象HEAD | `6a3f7bc` |
| コミット数 | 7 |
| 変更ファイル数 | 56 |
| 差分規模 | 約10,456行追加、927行削除 |

この文書ではコミット済みのブランチ差分を中心に説明する。
作成時点では、この文書以外の未コミット変更として `src/rikka/config.py` の `DATA_DIR` を
`input/sensor_data/nosidestep_1turn_3turn` へ切り替える未コミット変更が1件ある。

## 3. 全体の処理フロー

現在の通常PDRとparticle filterは、入力処理とPDRステップ準備を共有する。

```text
Accelerometer.csv / Gyroscope.csv
        │
        ├─ 時刻・数値・ジャイロbiasの検証
        ├─ センサー平滑化、重力軸、水平加速度の生成
        ├─ ステップ検出
        └─ prepare_pdr_steps()
              ├─ body heading
              ├─ motion heading
              ├─ 横歩き・旋回evidence
              ├─ 運動状態posterior
              └─ 歩幅と不確かさ
                    │
             ┌──────┴──────┐
             │             │
         通常PDR       particle filter
       軌跡を積算       地図制約・粒子伝播
                           ├─ ESS再標本化
                           ├─ map-aware recovery
                           ├─ checkpoint replay
                           └─ 代表経路選択
```

通常PDRとparticle filterで方位や歩幅を別々に再推定せず、
`prepare_pdr_steps()` の結果を共通入力として使用するようにした。

## 4. PDR・センサー処理の変更

### 4.1 体の向きと移動方向を分離

従来はジャイロ方位と加速度方位が混ざりやすかったため、次の役割を分離した。

- `body_heading`: ジャイロを中心に推定した体・端末の向き
- `motion_heading`: 水平加速度から得た実際の移動方向候補
- `movement_type`: センサー観測上の運動分類
- `trajectory_movement_type`: 最終的に軌跡へ反映した運動分類

これにより、体の向きを保ったまま横へ移動するサイドステップを表現できるようになった。

### 4.2 横歩き判定を単発判定から区間判定へ変更

横方向変位が一度だけ大きくなった歩を即座に横歩きとせず、同方向のevidenceが
連続するクラスタを確認してから確定するようにした。

- 横方向/前方向比率: `1.2`
- 最小横変位: `0.03 m`
- 標準平滑化: `clustered`
- 横歩き歩幅倍率: `0.8`
- 横歩き疑いは標準では前進扱い

旋回付き横歩きでは、クラスタ平滑化で横歩きを抑制しても生の旋回情報を失わないようにした。

### 4.3 動的な身体方位と運動区間復号

端末だけが回転した場合と、歩行者自身が旋回した場合を分けるため、次を追加した。

- 高信頼区間から身体方位を更新する処理
- 低信頼な横歩き誤検出を抑える処理
- 運動状態の区間復号
- 移動軸の `θ` / `θ + π` の方向曖昧性を保持するrobust方式

robust方式は一部データを改善したが、全体中央値を悪化させたため実験機能に留め、
標準設定にはしていない。

### 4.4 適応PDRを追加

`adaptive_estimator.py` を追加し、各歩について次を確率状態として逐次更新するようにした。

- 前進、左横歩き、右横歩き、旋回の状態確率
- headingの平均と不確かさ
- 前進・横歩きの局所歩幅倍率
- 端末と身体の方位差

`causal` は現在までの観測だけを使い、`offline` は記録全体で状態確率を後向き平滑化する。
最悪条件を含む評価では `causal` が安定したため、標準設定とした。

### 4.5 時刻・入力処理を堅牢化

レビューで見つかった次の問題を修正した。

- 歩幅の二重積分を固定100 HzではなくCSVの実時刻で計算
- 非有限、重複、逆行する明示時刻を早期にエラー化
- 非有限な初期方向、身長、ジャイロbias、PFパラメータを拒否
- 情報量が少なく方向確率が同率の場合は入力方位を優先
- 初期端末向き推定を高信頼な前進観測に限定

## 5. Particle filterの変更

### 5.1 粒子状態を拡張

粒子ごとに位置だけでなく、次の状態を保持するようにした。

- heading drift
- recovery用の方位状態
- 永続的な歩幅倍率
- 運動状態
- 祖先粒子と累積系列スコア

歩幅倍率は初期値をばらつかせ、観測と地図制約を通して学習し、再標本化後にも多様性を戻す。

### 5.2 地図制約を強化

終点画素だけでなく、1歩の始点から終点までの全画素を検査し、壁抜けとマップ外移動を拒否する。

particle処理の開始前には次も検証する。

- フロアマップが存在し、読み込める画像であること
- グレースケールへ正規化できること
- 原点がマップ内の歩行可能画素であること

### 5.3 ESSに基づく再標本化

毎歩無条件に再標本化せず、有効サンプルサイズ（ESS）が閾値を下回った場合だけ実行する。
再標本化後はheadingと歩幅倍率へ小さなノイズを加え、粒子の縮退を抑える。

### 5.4 map-aware recoveryとcheckpoint replay

有効粒子が不足した場合、地図上で合法な局所方位・歩幅候補を生成して復旧する。
現在位置から復旧できない場合は、直近の健全なcheckpointへ戻り、複数歩を再生する。

レビュー後、次の状態整合性も修正した。

- replay後の運動状態をcheckpointの親粒子から復元
- recoveryコストを累積系列スコアへ反映
- replayされた過去ステップの診断値を更新
- 診断歩幅を推定式ではなく実際の粒子変位から計算

重要な判断として、地図回避で選ばれた角度は恒久的なセンサー方位補正にはしていない。
恒久化すると局所的な壁回避角が後続全歩へ残り、終盤の誤方向収束を増やしたため、
減衰する一時的なheading driftとして保持している。

### 5.5 運動状態尤度を弱く重みへ反映

運動状態の予測尤度をそのまま粒子重みへ掛けると、一部seedだけが極端に悪化した。
複数データ・複数seed比較の結果、指数 `0.1` でtemperingして弱く反映する設定を採用した。

### 5.6 代表軌跡の選択

従来の到達可能な重み付き平均経路に加えて、単一粒子の完全な祖先系列を復元する
`sequence` 方式を追加した。

ただし常にsequenceを採用するのではなく、センサー上の旋回根拠がない持続反転が
従来経路より減る場合だけ採用し、それ以外は従来経路へ戻すguardを設けた。

## 6. 終端反転の診断と複数計測の統合

最後の曲がりで逆方向へ進む問題は、1歩で180度反転する場合だけでなく、
地図上の別の合法な廊下へ滑らかに入る場合がある。そのため次の指標を追加した。

- `terminal_direction_error_deg`
- `terminal_progress_cosine`
- `terminal_opposed_fraction`
- `terminal_direction_failure`

単一計測のセンサーと対称な地図だけでは分岐を一意に決められない場合があるため、
複数計測から代表軌跡を作る処理も追加した。

1. 壁交差、recovery failure、持続反転がある候補を除外する。
2. 計測・方式を等重みにして多数派の終端方向を求める。
3. 多数方向から90度を超えて外れる候補を除外する。
4. 残った候補からmedoidを選ぶ。

正解軌跡は代表選択には使わず、選択後の精度評価だけに使用する。

## 7. Particle filterの責務分割

巨大化していた `particle_filter.py` を互換facadeとして残し、内部処理を分割した。

| ファイル | 主な役割 |
|---|---|
| `particle/models.py` | 粒子診断データ型 |
| `particle/diagnostics.py` | ESS、分散、recoveryなどの診断生成 |
| `particle/resampling.py` | ESS計算と再標本化 |
| `particle/motion.py` | 運動状態、方位、尤度計算 |
| `particle/map_constraints.py` | 座標変換、壁・マップ外判定 |
| `particle/paths.py` | 祖先経路復元と代表軌跡選択 |
| `particle/recovery.py` | recovery候補とcheckpoint replay |
| `particle/runner.py` | particle filter全体の実行制御 |
| `particle/plotting.py` | 軌跡描画とアニメーション保存 |

既存の `rikka.analyze.particle_filter` importは維持している。
PDR内部処理への依存は `pdr/particle_api.py` をbridgeとして利用する。

## 8. CLI・出力・設定の変更

### 標準設定

| 設定 | 現在の標準値 | 意味 |
|---|---:|---|
| `MOTION_ESTIMATION` | `adaptive` | 確率的な運動状態・方位・歩幅推定 |
| `SMOOTHING_MODE` | `causal` | 現在までの観測だけを使用 |
| `PF_MOTION_PREDICTIVE_WEIGHT_POWER` | `0.1` | 運動状態尤度を弱く重みへ反映 |
| `PF_PATH_SELECTION` | `sequence` | 反転が減る場合だけ単一祖先系列を採用 |
| `SIDESTEP_SMOOTHING_METHOD` | `clustered` | 横歩きを連続区間で確定 |
| `SIDESTEP_LENGTH_SCALE` | `0.8` | 横歩きの歩幅倍率 |

CLIには方式、平滑化、seed、予測尤度指数、経路選択のオプションを追加した。

```sh
uv run rikka run --motion-estimation adaptive --smoothing causal
uv run rikka particle --pf-seed 42 \
  --motion-predictive-weight-power 0.1 \
  --pf-path-selection sequence
```

出力には次の診断CSVを追加・拡張した。

- `motion_posteriors.csv`: 運動状態、方位、歩幅、端末姿勢ずれの事後分布
- `direction_posteriors.csv`: robust方式の2方向確率と反転根拠
- `particle_diagnostics.csv`: ESS、粒子分散、歩幅倍率、recovery、経路選択
- `gyro_bias.csv`: ジャイロbias推定区間とfallback情報
- `step_length_observations.csv`: 各歩の歩幅観測と不確かさ

同名のセンサープロットが存在する場合は連番を付け、既存ファイルを上書きしない。

## 9. 評価・診断ツールの追加

再利用可能な検証コードを `agent/` に集約した。

| ファイル | 用途 |
|---|---|
| `agent_evaluate_pf_ground_truth.py` | 正解軌跡に対するPFの複数seed評価 |
| `agent_evaluate_adaptive_pdr.py` | PDR方式の複数計測比較 |
| `agent_benchmark_pdr_pf_methods.py` | PDR/PF、方式、seed、設定の横断比較 |
| `agent_diagnose_heading_reversal.py` | 持続反転と終端逆走の診断 |
| `agent_build_consensus_trajectory.py` | 複数候補から代表medoid軌跡を作成 |

`.agents/skills/` には、PDR変更検証とparticle filter評価の標準手順を追加した。
`agent/EXPERIMENT_LOG.md` にはEXP-001〜EXP-024として、採用案だけでなく失敗案、
実行条件、seed、指標、再検証条件も残している。

## 10. 精度評価結果

共通の正解軌跡を持つ5計測と、PF seed `[0, 1, 2, 10, 42, 100]` を使用した。

| 対象 | 条件 | RMSE中央値 | 最大RMSE | 壁交差 | recovery failure |
|---|---|---:|---:|---:|---:|
| 通常PDR | adaptive-causal、5計測 | 3.6224 m | 7.8383 m | - | - |
| PF | adaptive-causal、power 0.1、guard付きsequence、30実行 | 2.7957 m | 9.7266 m | 0 | 0 |
| PF標準データ | 同設定、6 seed | 0.8584 m | 2.5363 m | 0 | 0 |

標準データのPFでは6 seedすべてで終端方向failureが0だった。

### 残っている精度上の課題

地図上で合法な複数経路があり、センサー方位が分岐前にドリフトした場合は、
壁交差やrecovery failureがなくても誤った廊下へ収束する。

- data5: 全6 seedで終端方向failure
- data8: 全6 seedで終端方向failure
- data6: seed 0で終端方向failure

この問題に対する「終端だけ反転する後処理」は、別の場所へ人工的なUターンを作り、
全体形状を悪化させたため採用していない。現状は複数計測コンセンサスで代表軌跡から
外す方針としている。

## 11. 試したが標準採用しなかった方法

| 方法 | 不採用・限定採用の理由 |
|---|---|
| recovery角の恒久的なheading補正 | 局所回避角が後続全歩へ残り、終端誤方向とRMSEを大幅に悪化させた |
| posterior最尤状態のheadingをPDRへ強制 | 既に良いセンサー方位を90度単位で置き換え、5計測の精度を悪化させた |
| robust方向復号の標準化 | 一部データは改善したが、全体中央値を悪化させた |
| recovery枝quotaの標準有効化 | 誤った枝も保護し、spreadと一部seedを悪化させた |
| 終端軌跡の強制反転 | 別位置に人工的なUターンを作り、形状評価を悪化させた |
| 終点誤差だけで方式を選択 | 終点が近くても途中軌跡や終端方向が誤るため不十分だった |

## 12. テストと品質確認

方向曖昧性、入力検証、recovery、checkpoint replay、sequence経路、PDR状態整合、
終端反転を対象にした回帰テストを追加した。

最終確認結果:

- `uv run pytest`: 159件成功
- `uv run ruff check`: 成功
- `uv run mypy src/`: 成功
- `uv run pre-commit run --all-files`: 成功
- `uv build`: wheel / sdist生成成功
- 通常PDRとPFのCLI実データ実行: 成功

## 13. コミットごとの概要

| Commit | 内容 |
|---|---|
| `fcb244a` | 横歩き判定、PF診断、地図制約、recoveryの基礎を追加 |
| `778133e` | PDR/PF検証Skillとリポジトリ作業規約を追加 |
| `60264e1` | PF正解軌跡評価、実験ログ、運動観測の分離を追加 |
| `b848622` | 経路枝を保つ再標本化、身体方位、運動区間復号を追加 |
| `6c91440` | adaptive PDRと歩幅・方位・運動状態の確率推定を追加 |
| `c5070c3` | particle filterを責務別モジュールへ分割 |
| `6a3f7bc` | 横断benchmark、反転診断、レビュー修正、回帰テストを追加 |

## 14. 再検証コマンド

```sh
# 回帰テスト
uv run pytest
uv run ruff check
uv run mypy src/

# 標準PFの6 seed評価
MPLBACKEND=Agg uv run python agent/agent_evaluate_pf_ground_truth.py

# 5計測のPDR/PF横断評価
MPLBACKEND=Agg uv run python agent/agent_benchmark_pdr_pf_methods.py \
  --pdr-methods adaptive-causal \
  --pf-methods adaptive-causal \
  --seeds 0 1 2 10 42 100 \
  --motion-predictive-weight-powers 0.1 \
  --pf-path-selections sequence
```

方式や既定値を変更するときは、単一seedの見た目だけで判断せず、5計測の最大RMSE、
seed間ばらつき、壁交差、recovery failure、終端方向failureを同時に確認する。
