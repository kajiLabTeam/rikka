# Rikka 試行錯誤・実験ログ

PDR と particle filter の改善で、過去に試した仮説、条件、結果、採否を残す。
新しい調査を始める前にこのファイルを検索し、同じ条件の比較を理由なく繰り返さない。

## 記録ルール

- 新しい仮説、パラメータ比較、実データ評価を行ったら、成功・失敗にかかわらず1件追加する。
- 単なる既存テストの再実行は記録せず、新しい知見や設計判断が得られた場合に記録する。
- 「確認できた事実」と「推測」を分け、数値が残っていない結果を定量的に断定しない。
- 入力データ、正解軌跡、フロアマップ、主要パラメータ、seed、実行コマンドを再現できる粒度で書く。
- 画像やCSVを `output/` へ保存した場合も、このファイルには主要指標と結論を転記する。
- 未コミットの実装は、評価が終わるまで「採用」と記録しない。
- 過去の結論を再検証する場合は、再検証が必要になった変更点と過去エントリIDを書く。

## 新規エントリのテンプレート

```markdown
### EXP-XXX: 短い題名

- 日付:
- 状態: 進行中 / 採用 / 見送り / 保留
- 関連箇所・commit:
- 仮説:
- 入力:
- 比較条件:
- seed:
- 実行コマンド:
- 指標・観察結果:
- 結論:
- 採用内容:
- 再検証する条件:
```

## 先に確認する既知事項

- particle filter の良否を単一seedだけで判断しない。同じseedは再現確認、複数seedは安定性評価に使う。
- 同期していない正解軌跡とセンサー軌跡を行番号で直接比較しない。開始点を合わせ、正規化弧長で形状比較する。
- `1turn_rightsidestep_3turn_leftsidestep5`〜`8` は無印データと同じルート・歩行手順の反復計測であり、`walk_trace (3).csv` を共通の正解軌跡として扱う。時刻・歩数は同期していないため、各歩ラベルではなく正規化弧長上のルート形状を正解とする。
- 通常 PDR と particle filter で heading や step length を別々に推定しない。`prepare_pdr_steps()` の結果を共有する。
- 単発の強い横歩き evidence を、直ちに確定横歩きとして軌跡へ反映しない。連続クラスタと横変位を確認する。
- `prewalk_robust` がすべてのデータで最良とは限らない。既知の曲がりがあるデータでは手動biasとの比較を行う。
- 実験前から存在する `input/` と `output/` を削除しない。

## 実験一覧

| ID | 試行 | 状態 | 現在の判断 |
|---|---|---|---|
| EXP-001 | 水平加速度の移動方向補正 | 採用 | 初期前進ステップから固定ずれを推定し、無効化も可能にする |
| EXP-002 | 横歩き閾値と単発抑制 | 採用 | 比率と最小横変位を設定可能にし、単発候補を平滑化する |
| EXP-003 | 横歩き疑いの方位ガード比較 | 一部採用 | 疑い候補は標準では前進扱いにし、確定横歩きと分離する |
| EXP-004 | ジャイロbias推定方式 | 保留併用 | `prewalk_robust` を標準にし、問題データではmanualと比較する |
| EXP-005 | PDR処理の分割とPF共有 | 採用 | facade互換性を保ち、`prepare_pdr_steps()` とbridgeを共有点にする |
| EXP-006 | particle filter の乱数seed | 採用 | 固定seedで回帰を再現し、複数seedで安定性を評価する |
| EXP-007 | clustered横歩き判定 | 採用 | 同方向の連続evidenceと横変位を満たす区間だけ確定する |
| EXP-008 | PFの地図制約・復旧・診断 | 採用・継続評価 | 壁遷移、ESS、recovery、歩幅倍率を診断し、正解軌跡と複数seed比較する |
| EXP-009 | 平滑化で消える旋回付き横歩き | 採用 | 横歩きを抑制しても生の旋回情報は保持する |
| EXP-010 | 符号非依存の横歩きcluster | 見送り | 終点だけ改善し途中形状を悪化させるため採用しない |
| EXP-011 | PF運動状態の方位・歩幅反映範囲 | 一部採用 | 校正不十分かつ確定cluster内だけ状態別運動を反映する |
| EXP-012 | 行き止まりrecoveryの停止・後退候補 | 一部採用 | 停止候補は見送り、局所候補全滅時の後退候補を採用する |
| EXP-013 | 端末方位・身体方位候補・移動軸の観測分離 | 採用 | 三者を分離保持し、後段の区間復号と動的方位推定で使う |
| EXP-014 | 区間復号と動的身体方位 | 採用・継続評価 | 高信頼区間の補完と低校正偽陽性抑制を統合し、端末yawの過剰反映を抑える |
| EXP-015 | PF recoveryの経路枝quota | 実験機能のみ・既定無効 | 安全性は維持したが誤枝保護でspreadと一部seedが悪化した |
| EXP-016 | 横歩き歩幅補正 | 暫定採用・再検討 | 固定倍率0.8を採用したが、5〜8も共通正解ルートと確定したためPF精度を再検討する |
| EXP-017 | データ7・8の共通正解ルート再評価 | 調査完了・修正候補あり | 横歩き開始の方位跳びと区間別歩幅誤差を分離し、固定倍率だけでは直せないと確認した |

## 過去の試行詳細

### EXP-001: 水平加速度の移動方向補正

- 日付: 2026-06-16
- 状態: 採用
- 関連箇所・commit: `f0fd1d0`
- 仮説: 水平加速度から得た移動方向には端末姿勢由来の固定ずれがあり、初期前進区間から補正できる。
- 比較条件: 自動補正と補正なし。
- 指標・観察結果: 回帰テストを追加した。過去の定量値は保存されていない。
- 結論: 初期前進ステップから補正角を推定する方式を採用した。
- 採用内容: `motion_heading_correction` を自動推定し、`none` で無効化できるようにした。
- 再検証する条件: 端末装着方向や初期歩行パターンが異なるデータを追加したとき。

### EXP-002: 横歩き閾値と単発抑制

- 日付: 2026-06-22
- 状態: 採用
- 関連箇所・commit: `fb10861`
- 仮説: 横方向/前方向比だけでは微小変位を誤検出するため、最小横変位と時系列平滑化が必要。
- 比較条件: `SIDESTEP_LATERAL_RATIO`、`SIDESTEP_MIN_LATERAL_DISPLACEMENT_M`、平滑化なし、単発抑制。
- 指標・観察結果: 閾値変更、無効値、単発候補の回帰テストを追加した。過去のデータ別指標は保存されていない。
- 結論: 比率と最小横変位を併用し、孤立した横歩き候補をそのまま確定しない。
- 再検証する条件: 歩幅、端末装着者、サンプリングレートが大きく変わるとき。

### EXP-003: 横歩き疑いの方位ガード比較

- 日付: 2026-07-01〜2026-07-15
- 状態: 一部採用
- 関連箇所・commit: `b2613ac`, `fcb244a`
- 入力: `ryuki_1turn_rightsidestep_3turn_leftsidestep3` と `...4`。
- 比較条件: 現行、疑い候補を前進扱い、負の前方変位を棄却、bodyとの差が90度超のmotionを棄却、疑い方位の変化制限。
- 実行コマンド: `MPLBACKEND=Agg uv run python agent/agent_verify_heading_fix_comparison.py`
- 指標・観察結果: 比較プロット、方位差、負の前方変位、疑い候補数を確認するスクリプトを作成した。比較時の数値は履歴に保存されていない。
- 結論: 確定しなかった単発候補を標準では `forward` として扱い、確定横歩きと分離した。
- 採用内容: `SIDESTEP_SUSPECT_MODE = "forward"`。確定横歩きではmotion headingを使用する。
- 再検証する条件: 単発の横歩きを正しく検出すべきデータが追加されたとき。

### EXP-004: ジャイロbias推定方式

- 日付: 2026-07-09以降
- 状態: 保留併用
- 関連箇所・commit: `2e75a3f`
- 比較条件: `prewalk_robust`、`initial_robust`、`quietest`、`manual`。
- 指標・観察結果: 静止区間選択とfallbackの回帰テストを追加した。`1turn_rightsidestep_3turn_leftsidestep2` は `prewalk_robust` で曲がりやすい既知事例。
- 結論: 標準は `prewalk_robust` を維持するが、全データに対する万能な設定とは扱わない。
- 採用内容: 問題データでは `--gyro-bias-method manual --gyro-bias 0.002` を比較対象にする。
- 再検証する条件: 記録開始直後に静止区間がないデータ、歩行開始検出が不安定なデータ。

### EXP-005: PDR処理の分割とparticle filterとの共有

- 日付: 2026-07-10
- 状態: 採用
- 関連箇所・commit: `d2928ff`, `f64ba27`
- 仮説: 1ファイルに集中したPDR処理とPF側の重複推定は、変更時の不整合を生みやすい。
- 指標・観察結果: 処理をセンサー、bias、step、heading、sidestep、trajectory、outputへ分割し、回帰テストを維持した。
- 結論: `rikka.analyze.pdr` を互換facadeとして残し、共有ステップは `prepare_pdr_steps()` で一度だけ作る。
- 採用内容: PFからPDR内部を使う場合は `pdr/particle_api.py` をbridgeにする。
- 再検証する条件: facadeの公開import、共有データ型、PF入力を変更するとき。

### EXP-006: particle filter の乱数seed

- 日付: 2026-07-10
- 状態: 採用
- 関連箇所・commit: `7660ef6`
- 仮説: PFの回帰を単一実行で比較すると、乱数差と実装差を区別できない。
- 比較条件: 同一seedの再実行と異なるseedの実行。
- 指標・観察結果: 同じseedで同じ粒子軌跡になる回帰テストを追加した。
- 結論: `--pf-seed` を回帰再現用に使い、精度評価は複数seedで行う。
- 再検証する条件: 乱数ストリームの分割、resampling順序、particle数を変更するとき。

### EXP-007: clustered横歩き判定

- 日付: 2026-07-15
- 状態: 採用
- 関連箇所・commit: `fcb244a`
- 仮説: 同方向のevidenceが連続する区間は確定横歩きとして信頼できるが、孤立候補は誤検出しやすい。
- 比較条件: `none`、`isolated`、`clustered`。クラスタ内の1歩の隙間、方向一致、横変位強度も比較。
- 指標・観察結果: クラスタ、bridge gap、方向不一致、強い単発候補、軌跡用movement typeの回帰テストを追加した。
- 結論: `clustered` を標準にし、横歩きevidenceが2歩以上かつクラスタ横変位が閾値を満たす場合に確定する。
- 採用内容: 確定しない強い単発候補は、標準設定では軌跡上 `forward` にする。
- 再検証する条件: 1歩だけの実横歩きを含む評価データを整備したとき。

### EXP-008: PFの地図制約・復旧・診断

- 日付: 2026-07-15
- 状態: 採用・継続評価
- 関連箇所・commit: `fcb244a`
- 仮説: PFの最終軌跡だけでは退化や偶然の成功を判断できず、壁遷移、ESS、粒子多様性、recovery過程の記録が必要。
- 入力: `1turn_rightsidestep_3turn_leftsidestep`、対応する正解軌跡、フロアマップ。
- 比較条件: seed `[0, 1, 2, 10, 42, 100]`、方位・歩幅ノイズ、永続stride scale、ESS再標本化、map-aware recovery。
- 実行コマンド: `MPLBACKEND=Agg uv run python agent/agent_evaluate_pf_ground_truth.py`
- 指標・観察結果: RMSE、終点誤差、推定距離、壁交差、recovery failure、checkpoint replay、particle spreadを出力できるようにした。
- 結論: `wall_crossings == 0` と `recovery_failures == 0` を必須条件とし、精度値はseed別・中央値・最大値で判断する。
- 採用内容: `particle_diagnostics.csv` と複数seed評価スクリプトを追加した。
- 再検証する条件: 地図、起点、歩幅事前分布、motion state、recovery方式を変更するとき。

### EXP-009: 平滑化で消える旋回付き横歩き

- 日付: 2026-07-15
- 状態: 採用
- 関連箇所・commit: `190/particle-filter-sidestep` の未コミット差分、`src/rikka/analyze/pdr/sidestep.py`、`src/rikka/analyze/particle_filter.py`
- 仮説: 横歩きclusterが不成立でも、生の分類が `turning_sidestep_*` なら端末の旋回情報まで `forward` に落とすべきではない。
- 入力: `input/sensor_data/1turn_rightsidestep_3turn_leftsidestep5`〜`...8`。同じルートを反復計測したデータとして、共通の正解軌跡 `input/correct_path/1turn_rightsidestep_3turn_leftsidestep/walk_trace (3).csv` を使用する。
- 比較条件: clustered平滑化で未確定の `turning_sidestep_*` を `forward` にする従来処理と、生の旋回付き分類を保持する処理。通常PDRを先に比較し、その共有ステップをPFへ渡した。
- seed: 通常PDRはseedなし。PFは `[0, 1, 2, 10, 42, 100]`。
- 実行コマンド:
  - `MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run rikka run -d input/sensor_data/1turn_rightsidestep_3turn_leftsidestep5 --no-plot`（末尾を6〜8へ変更して実行）
  - `MPLCONFIGDIR=/tmp/rikka-mpl MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python agent/agent_evaluate_pf_ground_truth.py --data-dir input/sensor_data/1turn_rightsidestep_3turn_leftsidestep5 --seeds 0 1 2 10 42 100`（末尾を6〜8へ変更して実行。実験時のスクリプト配置は `scripts/`、記録時点では `agent/`）
- 指標・観察結果:
  - データ8はジャイロ上で27、42、62、77歩付近に旋回があったが、従来は42、62、77歩の生分類 `turning_sidestep_*` が軌跡用分類で `forward` になっていた。
  - 旋回保持後のデータ8通常PDRでは、この3歩が `turning_sidestep_*` として残った。通常PDR終点は従来 `[3.99, 3.72]` m、変更後 `[3.08, 4.24]` mだった。終点だけでなく、比較画像上で旋回後の区間形状も確認した。
  - データ5の通常PDR終点は `[10.65, 9.18]` mから `[10.03, 9.53]` m、データ6は `[-16.98, -2.99]` mから `[-17.12, -2.00]` mへ変化し、データ7は実質不変だった。
  - 最終PFの全seedで `wall_crossings == 0`、`recovery_failures == 0` を確認した。
- 結論: 横歩き確定の可否と旋回有無は別の情報として扱う。横歩きを抑制しても、閾値以上のyawを伴う生の `turning_sidestep_*` は消さない。
- 採用内容: 平滑化で横歩きを抑制する際の移動タイプを共通関数で決め、生の旋回付き分類を保持する。PF運動観測も平滑化後だけでなく生分類と `yaw_delta` から旋回尤度を作る。
- 再検証する条件: 端末だけを大きく回した非歩行旋回データ、旋回しながら横歩きしないデータ、反復ごとの歩ラベルや旋回境界を追加したとき。

### EXP-010: 符号非依存の横歩きcluster

- 日付: 2026-07-15
- 状態: 見送り
- 関連箇所・commit: 実験用差分のみ。最終コードから削除済み。
- 仮説: 1歩ごとの水平加速度積分は歩行周期で横変位の符号が反転するため、横歩きの存在を絶対横変位でcluster化し、左右方向をcluster確定後に決めればデータ8の欠落区間を復元できる。
- 入力: `1turn_rightsidestep_3turn_leftsidestep5`〜`...8`。誤検出確認として `nosidestep_1turn_3turn` と `ryuki_nosidestep_1turn_3turn` も実行した。
- 比較条件: 従来の同方向evidenceと1歩bridgeに対し、最大2歩の弱い区間と反対符号のevidenceを接続し、横変位の絶対値でcluster強度を判定した。
- seed: 通常PDR比較のためseedなし。
- 実行コマンド: `MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run rikka run -d input/sensor_data/1turn_rightsidestep_3turn_leftsidestep8 --no-plot`。同形式で5〜7と横歩きなし2データも実行した。
- 指標・観察結果:
  - データ8の32歩目は横変位 `+0.125` m、35歩目は `-0.091` mで、間の33、34歩も弱い横変位を持っていた。この案では32〜35歩が1つの横歩きclusterになった。
  - データ8の通常PDR終点は `[3.08, 4.24]` mから `[3.29, 0.71]` mへ共通正解軌跡の終点に近づいた。
  - しかし軌跡比較では、最初の横歩き区間以降が大きく内側へ曲がり、途中形状が旋回保持だけの案より悪化した。左右方向の根拠も歩ごとに一致しなかった。
  - `nosidestep_1turn_3turn` は横歩きcluster 0だった。`ryuki_nosidestep_1turn_3turn` は端末姿勢校正信頼度が約0.282で、生分類自体が横歩きへ偏っており7 cluster、横歩き系76歩になった。この問題は符号非依存clusterだけでは解決できなかった。
- 結論: 終点改善だけでは採用できない。符号反転を許す場合も、移動軸の一貫性と左右方向を別の観測で保証する必要がある。
- 採用内容: なし。同方向evidence、最大1歩bridge、符号付きcluster強度の従来条件へ戻した。
- 再検証する条件: 横歩き区間ごとの正解ラベル、端末座標に依存しない移動軸推定、左右方向のcluster単位正解を用意できたとき。

### EXP-011: PF運動状態の方位・歩幅反映範囲

- 日付: 2026-07-15
- 状態: 一部採用
- 関連箇所・commit: `190/particle-filter-sidestep` の未コミット差分、`src/rikka/analyze/pdr/models.py`、`src/rikka/analyze/pdr/sidestep.py`、`src/rikka/analyze/particle_filter.py`
- 仮説: forward・左右横歩き・旋回の状態をPF粒子ごとに持ち、PDRの運動観測と状態遷移を組み合わせれば、低信頼度データでも確定横歩きclusterをPFへ反映できる。
- 入力: `1turn_rightsidestep_3turn_leftsidestep5`〜`...8`、フロアマップ、同一ルートの共通正解軌跡。
- 比較条件:
  - 従来: PDRの確定方位・歩幅を全粒子へ共通適用。
  - 案A: 全歩で粒子状態別のbody/motion headingと歩幅倍率を適用。
  - 最終案: 運動状態は全歩で追跡するが、状態別方位・歩幅は校正信頼度 `< 0.85` かつ `sidestep_cluster_id` がある歩だけに適用。
  - 確定横歩きclusterは校正信頼度 `>= 0.5` のとき横歩き尤度を強めた。
- seed: `[0, 1, 2, 10, 42, 100]`。
- 実行コマンド: `MPLCONFIGDIR=/tmp/rikka-mpl MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python agent/agent_evaluate_pf_ground_truth.py --data-dir input/sensor_data/1turn_rightsidestep_3turn_leftsidestep7 --seeds 0 1 2 10 42 100`。同形式で5、6、8も実行した。
- 指標・観察結果:
  - データ7のPDRは34〜38歩と66〜70歩の計10歩を横歩きとしたが、従来PFの代表横歩き状態はseedごとに0〜3歩だった。
  - 案Aではデータ7の代表横歩き状態が9〜13歩になり、正解軌跡RMSEは `[4.50, 7.17, 8.24, 2.23, 3.10, 2.38]` m、中央値約3.80 mまで下がった。一方で最大値は8.24 mへ悪化し、seed 1の最大位置spreadは約7.89 mになった。
  - 案Aを校正信頼度の高いデータ5、6にも適用すると、それまで安定していたPDR方位を状態が上書きし、内側へのloopが増えたため全歩適用は見送った。
  - 最終案のデータ7代表横歩き状態は全seedで8〜9歩。RMSEは `[5.72, 5.78, 6.31, 5.80, 5.79, 6.24]` mで、従来の約5.68〜5.80 mに対する形状精度の明確な改善は確認できなかったが、状態診断とPDR確定clusterの不一致は解消した。
  - 最終6seedの正解軌跡RMSE中央値/最大値は、データ5が7.05/8.93 m、データ6が5.61/10.30 m、データ7が5.80/6.31 m、データ8が6.35/9.60 m。
- 結論: 低校正信頼度で確定したclusterをPF状態へ反映することには意味があるが、確率状態がcluster外の方位・歩幅を変更するとseed分岐を増やす。状態診断の改善と軌跡精度の改善を同一視しない。
- 採用内容: `StepMotionEvidence`、PF運動状態遷移、状態確率・entropy・遷移数の診断を追加する。状態別方位・歩幅の実反映は校正不十分かつ確定横歩きcluster内に限定する。
- 再検証する条件: 各歩の移動状態ラベル、低校正信頼度かつ横歩きなしの追加データを用意したとき。状態遷移確率を変える場合も6seed全体で再評価する。

### EXP-012: 行き止まりrecoveryの停止・後退候補

- 日付: 2026-07-15
- 状態: 一部採用
- 関連箇所・commit: `190/particle-filter-sidestep` の未コミット差分、`src/rikka/analyze/particle_filter.py`
- 仮説: 誤分岐の行き止まりで局所・90度候補が全滅した場合、停止または後退を許せば `failed_hold` を避け、次の歩で経路へ復帰できる。
- 入力: `1turn_rightsidestep_3turn_leftsidestep6`、seed 0。最終確認は5〜8の6seed。
- 比較条件:
  - 従来候補: `0, ±5, ±10, ±20, ±30, ±45` 度、必要時 `±60, ±90` 度。
  - 停止案: 全移動候補が無効なとき歩幅倍率0を最後に追加。
  - 後退案: 停止案を削除し、局所候補全滅時のturn gridへ `±135, 180` 度を追加。
- seed: 主比較はデータ6 seed 0。最終確認は `[0, 1, 2, 10, 42, 100]`。
- 実行コマンド: `MPLCONFIGDIR=/tmp/rikka-mpl MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python agent/agent_evaluate_pf_ground_truth.py --data-dir input/sensor_data/1turn_rightsidestep_3turn_leftsidestep6 --seeds 0`。採用後は5〜8を6seedで再実行した。
- 指標・観察結果:
  - 従来のデータ6 seed 0では62歩目が前進、`valid_count == 0`、`recovery_mode == "failed_hold"` になった。48〜60歩でもrecoveryが繰り返され、誤った枝の行き止まりへ収束していた。
  - 停止案はrecovery failureを0にしたが、停止を繰り返してrecovery 30回、推定距離64.68 m、RMSE 10.96 mとなった。失敗表示を隠すだけで経路から脱出できなかったため見送った。
  - 後退案ではデータ6 seed 0のrecovery failure 0、recovery 20回、推定距離74.28 m、RMSE 9.31 m、最大位置spread 5.03 mとなった。少なくとも停止せず、後退して分岐仮説を広げられた。
  - 最終評価では5〜8の全24実行で壁交差0、recovery failure 0だった。ただしrecoveryはデータ5で18〜20回、データ6で16〜21回、データ7で15〜24回、データ8で13〜20回と依然多い。
- 結論: 停止は有効な復旧ではない。明示的な後退候補は行き止まり脱出に有効だが、マップだけでは正しい分岐を一意に選べず、recovery多発自体は解消しない。
- 採用内容: fallback turn gridへ `±135, 180` 度を追加する。歩幅0のstationary recoveryは採用しない。
- 再検証する条件: 分岐履歴、旋回列、checkpoint深度、複数枝を保持するresamplingを変更したとき。特にrecovery回数と最大spreadを6seedで再評価する。

### EXP-013: 端末方位・身体方位候補・移動軸の観測分離

- 日付: 2026-07-15
- 状態: 採用（挙動非変更の基盤）
- 関連箇所・commit: `190/particle-filter-sidestep` の未コミット差分、`src/rikka/analyze/pdr/models.py`、`src/rikka/analyze/pdr/sidestep.py`、`src/rikka/analyze/pdr/trajectory.py`
- 仮説: 端末のyaw、身体方位候補、水平加速度由来の移動方位を1つの確定headingにすぐ潰さず保持すれば、後段の時系列判定で端末の曲がりと歩行方向の変化を分けて扱える。
- 入力: 単体回帰テストと `input/sensor_data/1turn_rightsidestep_3turn_leftsidestep5`。
- 比較条件: 既存の `StepHeading` / 軌跡を変更せず、`device_yaw_heading`、`body_heading_candidate`、`directed_motion_heading`、modulo pi の `motion_axis_heading` を `StepMotionObservation` として併記する。
- seed: 通常PDRと観測構築のためseedなし。
- 実行コマンド:
  - `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_pdr_regressions.py -q`
  - `UV_CACHE_DIR=.uv-cache uv run pytest -q`
  - `UV_CACHE_DIR=.uv-cache uv run mypy src/`
  - `UV_CACHE_DIR=.uv-cache uv run ruff check src/rikka/analyze/pdr/models.py src/rikka/analyze/pdr/sidestep.py src/rikka/analyze/pdr/trajectory.py src/rikka/analyze/pdr/particle_api.py src/rikka/analyze/pdr/__init__.py tests/test_pdr_regressions.py`
  - `MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run rikka run -d input/sensor_data/1turn_rightsidestep_3turn_leftsidestep5 --no-plot`
- 指標・観察結果:
  - 移動方位35度と215度が同一の `motion_axis_heading` になること、端末yaw 10度・身体方位候補25度・有向移動方位170度を別フィールドで保持することを回帰テストで確認した。
  - PDR回帰95件、全テスト108件、Ruff、Mypyが通過した。
  - データ5は91歩、終点 `[10.0287218855, 9.5333646379]` mで、EXP-009の旋回保持後の終点 `[10.03, 9.53]` mと一致した。`trajectory.csv`、`step_lengths.csv`、`step_headings.csv` はヘッダー込み92行で整合した。
- 結論: 観測の分離を軌跡振る舞いを変えずに追加できた。これは横歩き区間デコーダと動的な端末−身体オフセット推定の入力に使う。
- 採用内容: `StepMotionObservation`、`build_step_motion_observations()`、`PreparedPdrSteps.motion_observations` を追加し、PFからは `pdr/particle_api.py` を通して利用可能にした。
- 再検証する条件: 観測フィールドを追加するとき、区間デコーダが移動軸の向きを確定するとき、PFの粒子状態へ反映するとき。

### EXP-014: 区間復号と動的身体方位の統合

- 日付: 2026-07-16
- 状態: 採用・継続評価
- 関連箇所・commit: `190/particle-filter-sidestep` の未コミット差分、`pdr/motion_decoder.py`、`pdr/body_heading.py`、`pdr/motion_refinement.py`、`pdr/trajectory.py`
- 仮説: 横歩きを1歩ごとの符号ではなく modulo pi の移動軸と連続区間で復号し、端末yawと身体方位の時変差を別状態で推定すれば、端末を曲げた量が進行方向へ直接反映されにくくなる。
- 入力: `1turn_rightsidestep_3turn_leftsidestep5`〜`8`、`nosidestep_1turn_3turn`、`ryuki_nosidestep_1turn_3turn`、同一ルートの共通正解軌跡、フロアマップ。正解軌跡はルート形状の正解とするが、時刻非同期のため歩境界は正解扱いしていない。
- 比較条件: 従来cluster / 区間decoderのshadow / decoder→動的body offset→体軸再射影→再decode。最終統合は、高信頼decoderで欠落を補完し、校正信頼度 `< 0.45` のときだけdecoder-forwardでlegacy偽陽性を抑制した。高校正のlegacy clusterは保持した。
- seed: 通常PDRはseedなし。PFは `[0, 1, 2, 10, 42, 100]`。
- 実行コマンド:
  - `MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python /private/tmp/rikka_motion_baseline.py`
  - `MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python /private/tmp/rikka_plot_refinement.py`
  - `MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run rikka run -d input/sensor_data/1turn_rightsidestep_3turn_leftsidestep5 --no-plot`（末尾を6〜8へ変更）
  - `MPLCONFIGDIR=/tmp/rikka-mpl MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python agent/agent_evaluate_pf_ground_truth.py --data-dir input/sensor_data/1turn_rightsidestep_3turn_leftsidestep5 --seeds 0 1 2 10 42 100`（末尾を6〜8へ変更）
- 指標・観察結果:
  - 通常PDRの横歩き区間はデータ5が34〜42、69〜75、82歩、6が37〜43、48、68〜70、72〜80歩、7が34〜40、66〜70歩、8が32〜33、42、62、68〜70、77歩となった。データ8の従来欠落区間32〜33歩を補完し、データ7の既存良好clusterを保持した。
  - `ryuki_nosidestep_1turn_3turn` は従来の7 cluster・横歩き系76歩から0へ減少した。`nosidestep_1turn_3turn` も0を維持した。
  - 通常PDR終点の従来→変更後は、5 `[10.03, 9.53]`→`[7.17, 9.00]`、6 `[-17.12, -2.00]`→`[-16.41, 0.04]`、7 `[-0.91, 9.19]`→`[0.21, 8.62]`、8 `[3.08, 4.24]`→`[3.40, -1.03]` m。共通正解軌跡はほぼ始点へ戻るが、途中形状を隠さないよう終点だけでは合否を決めていない。
  - 比較画像ではデータ8の中盤の大きな膨らみが縮小し、横歩きなしデータの偽clusterが消えた。一方、データ8は共通正解軌跡RMSEの中央値が悪化した。
  - 最終PFのRMSE中央値/最大値は、5が6.03/6.79 m（EXP-011の7.05/8.93）、6が2.02/2.28 m（5.61/10.30）、7が5.01/6.22 m（5.80/6.31）、8が7.49/9.19 m（6.35/9.60）。全24実行で壁交差0、recovery failure 0。
  - recovery回数の中央値/最大値は5が17/21、6が20.5/23、7が7.5/16、8が19/21。
- 結論: 端末yawを身体方位と即断せず、区間で確定した移動モードから動的に端末−身体差を更新する方式を採用する。世界移動方位は二重補正せず、body headingと体軸射影だけを更新する。
- 採用内容: semi-Markov区間decoder、3歩因果窓・Huber残差・最大5度/歩の動的body offset、低校正fallback、診断CSV列、`motion_refinement=False` によるAPI比較経路。
- 再検証する条件: 5〜8の歩ラベルが得られたとき、1歩だけの実横歩き・端末だけを曲げた非旋回データを追加したとき、デコーダ閾値や更新上限を変えるとき。

### EXP-015: PF recovery候補の経路枝quota

- 日付: 2026-07-16
- 状態: 実験機能のみ保持・既定無効
- 関連箇所・commit: `src/rikka/analyze/particle_branches.py`、`src/rikka/analyze/particle_filter.py`、`agent/agent_evaluate_pf_ground_truth.py`
- 仮説: local recoveryで有効候補が1つでも見つかったときにturn候補を破棄せず、直進・左・右・後退族へ最低quotaを与えれば誤分岐の行き止まり前に別枝を残せる。
- 入力: データ5〜8、共通正解軌跡、フロアマップ。
- 比較条件: 従来のlocal早期returnと、local+turnの全有効候補を4族に分類し `ceil(sqrt(N))` quotaを与える方式。`--enable-recovery-branches` で後者を明示有効化した。
- seed: `[0, 1, 2, 10, 42, 100]`。
- 実行コマンド: `MPLCONFIGDIR=/tmp/rikka-mpl MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python agent/agent_evaluate_pf_ground_truth.py --data-dir input/sensor_data/1turn_rightsidestep_3turn_leftsidestep6 --seeds 0 1 2 10 42 100 --enable-recovery-branches`（末尾を5、7、8へ変更）。
- 指標・観察結果:
  - 合成open mapで20粒子を4族へ各5粒子残し、少数枝が即消滅しないことを確認した。
  - 有効時も全seedで壁交差0、recovery failure 0だった。
  - データ6のRMSE中央値/最大値は従来recovery 2.02/2.28 mに対し枝quota 4.14/8.14 m、最大spreadは1.36 mから4.90 mへ悪化した。
  - データ8は従来recoveryのRMSE中央値/最大値7.49/9.19 mに対し枝quota 9.19/10.83 m、最大spreadは1.09 mから5.30 mへ悪化した。
  - データ5の一部seedは大きく改善したが、6・8の誤枝保護とseed間ばらつきを相殺できなかった。
- 結論: 地図の壁通過可否だけで枝quotaを与えると、正枝と同時に誤枝も保護してspreadを増やす。経路枝保持は実装を残すが既定無効とする。
- 採用内容: 枝内systematic resampling、枝quota、recovery枝診断、`preserve_recovery_branches=False` の安全な既定。
- 再検証する条件: 旋回イベントの身体方位確率、分岐後数歩の区間尤度、または正解経路ラベルで誤枝を間引けるようになったとき。

### EXP-016: 横歩き歩幅補正

- 日付: 2026-07-16
- 状態: 暫定採用・再検討
- 関連箇所・commit: `src/rikka/config.py`、`src/rikka/analyze/pdr/sidestep.py`、`190/particle-filter-sidestep` の未コミット差分
- 仮説: 横歩きにも前進と同じWeinberg係数を使い倍率1.0を掛けると、4乗根で上下加速度振幅差が圧縮され、実際より長い横歩きとして軌跡が歪む。移動状態確定後に横歩き歩幅だけ縮小すれば改善できる。
- 入力: 共通正解軌跡に対応する無印データと`1turn_rightsidestep_3turn_leftsidestep5`〜`8`、横歩きなし2データ、フロアマップ。
- 比較条件: 固定倍率0.30〜1.0、全前進歩・直近8歩・直近12歩の中央値による適応cap、decoder confidence比例、cluster外単発旋回の追加縮小。PFは主要候補を6 seedで比較した。
- seed: PFは `[0, 1, 2, 10, 42, 100]`。
- 実行コマンド:
  - `PYTHONPATH=. MPLCONFIGDIR=/tmp/rikka-mpl MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python /tmp/rikka_step_length_experiment.py`
  - `PYTHONPATH=. MPLCONFIGDIR=/tmp/rikka-mpl MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python /tmp/rikka_pf_step_scale_eval.py`
  - `PYTHONPATH=. MPLCONFIGDIR=/tmp/rikka-mpl MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python /tmp/rikka_pf_exact_scale_eval.py`
- 指標・観察結果:
  - 倍率1.0では横歩き/前進の歩幅中央値比がデータ7で0.979、データ8で0.960だった。上下加速度振幅は横歩きの方が小さいが、Weinberg式の4乗根で歩幅差が小さくなっていた。
  - PDR正解軌跡RMSEは倍率1.0→0.8で、データ5が7.549→6.865 m、6が8.229→7.963 m、7が5.186→4.346 m、8が4.202→3.818 mとなった。無印データも2.804→2.656 mへ改善した。
  - 適応capは固定倍率を明確に上回らなかった。cluster外単発旋回だけを0.3へ縮める案は5〜8で改善したが、正解対応データを悪化させた。decoder confidenceは単発旋回の方が連続clusterより高く、歩幅倍率へ直接使えなかった。
  - 正解対応データのPFでは倍率0.8がRMSE中央値/最大値1.10/1.78 mで、倍率1.0の1.52/4.41 m、0.85の1.33/1.86 mより良かった。recovery中央値は7回、failureは0だった。
  - データ5〜8の倍率0.8 PFは全24実行で壁交差0、failure 0だった。正解軌跡RMSE中央値/最大値は5が5.63/7.83、6が3.33/11.59、7が5.74/9.90、8が7.83/9.44 mで、一部seedは倍率1.0より悪化した。
  - 横歩きなし2データは横歩き判定0で、補正による軌跡差は0だった。
- 結論: 強い縮小0.55〜0.60はPDR正解軌跡を最も縮めるが、PFのデータ6・7で不安定だった。適応式や単発例外より単純で、無印データのPDR/PFとデータ5〜8のPDRを同時に改善した固定0.8を採用した。ただし5〜8も同じ正解ルートであるため、PF指標悪化は未解決の精度問題として扱う。
- 採用内容: `SIDESTEP_LENGTH_SCALE = 0.8`。通常PDRとPFは `prepare_pdr_steps()` を通して同じ補正済み歩幅を共有する。
- 再検証する条件: 横歩きの実測距離、歩ごとの移動状態ラベルが得られたとき。倍率変更時は無印データと5〜8を同じ6 seedで再評価する。

### EXP-017: データ7・8の共通正解ルート再評価

- 日付: 2026-07-16
- 状態: 調査完了・修正候補あり
- 関連箇所・commit: `pdr/sidestep.py`、`pdr/motion_refinement.py`、`pdr/step_length.py`、`particle_filter.py`、`190/particle-filter-sidestep` の未コミット差分
- 仮説: データ7・8の異常は横歩き歩幅だけでなく、横歩き開始時の方位不連続と、同一記録内の区間ごとの歩幅変化を固定Weinberg係数で表現できないことから生じる。
- 入力: `1turn_rightsidestep_3turn_leftsidestep7`、`...8`、両データと同じルートの共通正解軌跡、既存PF出力 `20260716_090806_814339`、`20260716_100940_843472`、`20260716_101030_133947`。
- 比較条件: motion refinement有効/無効、旋回なし横歩き開始の方位差45/60/75度ガード、正解軌跡と推定軌跡の旋回間区間長比較。
- seed: 通常PDRはseedなし。既存PF出力は各実行時seedに従う。過去の6 seed集計はEXP-016を参照。
- 実行コマンド:
  - `PYTHONPATH=. MPLCONFIGDIR=/tmp/rikka-mpl MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python /tmp/rikka_diagnose_78.py`
  - `PYTHONPATH=. MPLCONFIGDIR=/tmp/rikka-mpl MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python /tmp/rikka_sidestep_onset_guard.py`
- 指標・観察結果:
  - 共通正解ルートをRDP 0.5 mで単純化した旋回間距離は約 `[24.35, 12.00, 13.05, 11.92, 11.13]` mだった。
  - データ7の推定区間距離は `[22.77, 11.58, 15.75, 11.16, 8.83]` m、データ8は `[22.79, 11.84, 16.63, 11.96, 7.58]` mだった。両方で3区間目が21〜27%過大、最終区間が21〜32%過小となり、単一の全体歩幅倍率では同時に補正できない。
  - データ7は最初の横歩き開始時に直前方位約154度から約73度へ、旋回なしで約81度跳んだ。2回目の横歩き開始にも同種の跳びがあり、大きな内側ループの主因になった。
  - 旋回なし横歩き開始で45度を超える方位跳びを直前方位へ戻す一時案は、PDR正解軌跡RMSEをデータ7で4.346→3.081 m、8で3.818→3.683 mへ改善した。データ5は不変、6は7.963→7.830 mだった。
  - motion refinement無効/有効のRMSEはデータ7が4.572/4.346 m、8が2.912/3.818 mだった。データ8の32〜33歩補完はルート正解基準では悪化しており、補完の採用条件を再検討する必要がある。
  - 既存PF画像ではデータ8が下側通路を往復し、データ7が外周を一周せず右縦通路へ早期復帰していた。3区間目の過大歩幅で旋回観測前に壁へ到達し、recoveryが地図上で通行可能な誤分岐を選ぶことと整合する。
- 結論: データ7は横歩き開始時の方位連続性、データ8は区間別歩幅変動と32〜33歩の補完条件が主要課題である。固定 `SIDESTEP_LENGTH_SCALE` の再調整だけでは解決しない。PFでは区間内歩幅倍率の下限・学習と、旋回列に整合するrecovery枝評価を一緒に検証する必要がある。
- 採用内容: 評価前提と診断結果のみ記録。製品コードへの方位ガード・PF歩幅状態変更は未採用。
- 再検証する条件: 横歩き開始ガードを本実装するとき、PF stride scale範囲やprocess noiseを変更するとき、データ8の32〜33歩を確定横歩きへ補完する条件を変更するとき。

## 進行中の試行

進行中の実装は、評価条件と結果が確定した時点でここへ追加する。
現在の未コミット差分から結論を推測して記載しない。
