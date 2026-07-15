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
- 入力: `input/sensor_data/1turn_rightsidestep_3turn_leftsidestep5`〜`...8`。同じ歩行手順の参考形状として `input/correct_path/1turn_rightsidestep_3turn_leftsidestep/walk_trace (3).csv` を使用したが、5〜8固有の正解軌跡ではない。
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
- 再検証する条件: 端末だけを大きく回した非歩行旋回データ、旋回しながら横歩きしないデータ、5〜8固有の正解軌跡を追加したとき。

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
  - データ8の通常PDR終点は `[3.08, 4.24]` mから `[3.29, 0.71]` mへ参考形状の終点に近づいた。
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
- 入力: `1turn_rightsidestep_3turn_leftsidestep5`〜`...8`、フロアマップ、同一手順のshape reference。
- 比較条件:
  - 従来: PDRの確定方位・歩幅を全粒子へ共通適用。
  - 案A: 全歩で粒子状態別のbody/motion headingと歩幅倍率を適用。
  - 最終案: 運動状態は全歩で追跡するが、状態別方位・歩幅は校正信頼度 `< 0.85` かつ `sidestep_cluster_id` がある歩だけに適用。
  - 確定横歩きclusterは校正信頼度 `>= 0.5` のとき横歩き尤度を強めた。
- seed: `[0, 1, 2, 10, 42, 100]`。
- 実行コマンド: `MPLCONFIGDIR=/tmp/rikka-mpl MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run python agent/agent_evaluate_pf_ground_truth.py --data-dir input/sensor_data/1turn_rightsidestep_3turn_leftsidestep7 --seeds 0 1 2 10 42 100`。同形式で5、6、8も実行した。
- 指標・観察結果:
  - データ7のPDRは34〜38歩と66〜70歩の計10歩を横歩きとしたが、従来PFの代表横歩き状態はseedごとに0〜3歩だった。
  - 案Aではデータ7の代表横歩き状態が9〜13歩になり、shape-reference RMSEは `[4.50, 7.17, 8.24, 2.23, 3.10, 2.38]` m、中央値約3.80 mまで下がった。一方で最大値は8.24 mへ悪化し、seed 1の最大位置spreadは約7.89 mになった。
  - 案Aを校正信頼度の高いデータ5、6にも適用すると、それまで安定していたPDR方位を状態が上書きし、内側へのloopが増えたため全歩適用は見送った。
  - 最終案のデータ7代表横歩き状態は全seedで8〜9歩。RMSEは `[5.72, 5.78, 6.31, 5.80, 5.79, 6.24]` mで、従来の約5.68〜5.80 mに対する形状精度の明確な改善は確認できなかったが、状態診断とPDR確定clusterの不一致は解消した。
  - 最終6seedのRMSE中央値/最大値は、データ5が7.05/8.93 m、データ6が5.61/10.30 m、データ7が5.80/6.31 m、データ8が6.35/9.60 m。これは5〜8固有の正解ではなくshape referenceに対する値である。
- 結論: 低校正信頼度で確定したclusterをPF状態へ反映することには意味があるが、確率状態がcluster外の方位・歩幅を変更するとseed分岐を増やす。状態診断の改善と軌跡精度の改善を同一視しない。
- 採用内容: `StepMotionEvidence`、PF運動状態遷移、状態確率・entropy・遷移数の診断を追加する。状態別方位・歩幅の実反映は校正不十分かつ確定横歩きcluster内に限定する。
- 再検証する条件: 5〜8固有の正解軌跡、各歩の移動状態ラベル、低校正信頼度かつ横歩きなしの追加データを用意したとき。状態遷移確率を変える場合も6seed全体で再評価する。

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

## 進行中の試行

進行中の実装は、評価条件と結果が確定した時点でここへ追加する。
現在の未コミット差分から結論を推測して記載しない。
