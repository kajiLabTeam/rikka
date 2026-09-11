# 現在の標準設定と切り替え可能な処理

値は [`src/rikka/common/config/__init__.py`](../../src/rikka/common/config/__init__.py) と [`cli/options.py`](../../src/rikka/cli/options.py) の伝播を確認したものです。
Python APIでは [`cli.commands.run()`](../../src/rikka/cli/commands.py#L81) の既定引数から同じ設定へ変換されます。

| 対象処理 | 現在の設定 | 切り替え候補 | 設定場所 | 参照箇所 | 評価結果 |
|---|---|---|---|---|---|
| 入力 | `hiroto/...leftsidestep` | `-d PATH` | [`DATA_DIR`](../../src/rikka/common/config/__init__.py#L21) | [`load_sensor_data`](../../src/rikka/common/lib/sensors.py#L51) / CLI | 現行入力のtruth評価は未確認 |
| 初期方向 | 90° | 任意有限値 | [`INITIAL_DIRECTION`](../../src/rikka/common/config/__init__.py#L34) | heading resolver | 単独比較なし |
| サンプリング率fallback | 100Hz | 定数変更のみ | [`SAMPLING_RATE`](../../src/rikka/common/config/__init__.py#L37) | time/step length | 実時刻列を優先 |
| gyro bias | `prewalk_guarded` | zero / prewalk_robust / initial_robust / quietest / manual | [`GYRO_BIAS_METHOD`](../../src/rikka/common/config/__init__.py#L53) | sensors/gyro_bias | EXP-026で最終構成改善 |
| bias guard | 0.003rad/s | 定数変更のみ | [`GYRO_BIAS_GUARD_MAX_ABS_RAD_S`](../../src/rikka/common/config/__init__.py#L54) | estimators | 6計測で採用 |
| 歩検出 | `peak` | paper_vertical_threshold | [`STEP_DETECTION_METHOD`](../../src/rikka/common/config/__init__.py#L87) | step_detection | 同一条件比較なし |
| peak | height 1.0、distance 50 | 定数変更のみ | `PEAK_*` | step_detection | 根拠の最新比較なし |
| 歩幅 | `weinberg` | forward | [`STEP_LENGTH_METHOD`](../../src/rikka/common/config/__init__.py#L127) | trajectory/step_length | 同一条件比較なし |
| 身長 | 1.68m | `--height-m` | [`USER_HEIGHT_M`](../../src/rikka/common/config/__init__.py#L101) | compute_weinberg_k | 個人値 |
| 方位方式 | `gyro_accel_motion` | gyro / accel_method1 / accel_method2 | [`HEADING_METHOD`](../../src/rikka/common/config/__init__.py#L134) | heading resolver | 方式単独比較なし |
| motion補正 | auto | none | settings/CLI | heading motion | EXP-001採用、現行表なし |
| forward方位 | body | motion | [`FORWARD_HEADING_SOURCE`](../../src/rikka/common/config/__init__.py#L139) | heading_policy | bodyが標準、同一現行表なし |
| 横歩き方位 | motion | body_lateral / blend | CLI既定 | heading_policy | 同一条件表なし |
| 横歩き比率 | 1.2 | 正のfloat | [`SIDESTEP_LATERAL_RATIO`](../../src/rikka/common/config/__init__.py#L151) | classification | 閾値単独表なし |
| 最小横変位 | 0.03m | 0以上float | 設定 | classification | 閾値単独表なし |
| 横歩き平滑化 | clustered | none / isolated | [`SIDESTEP_SMOOTHING_METHOD`](../../src/rikka/common/config/__init__.py#L153) | clustering | clustered採用 |
| suspect | forward | motion / body_lateral / blend | [`SIDESTEP_SUSPECT_MODE`](../../src/rikka/common/config/__init__.py#L140) | clustering/step_motion | forward標準 |
| 横歩き歩幅 | 0.8 | 定数変更のみ | [`SIDESTEP_LENGTH_SCALE`](../../src/rikka/common/config/__init__.py#L155) | step_motion/PF | EXP-016、暫定採用 |
| 旋回歩幅 | 0.3 | 定数変更のみ | [`TURNING_LENGTH_SCALE`](../../src/rikka/common/config/__init__.py#L156) | step_motion/PF | 単独比較なし |
| motion fusion | adaptive | legacy / robust | [`MOTION_ESTIMATION`](../../src/rikka/common/config/__init__.py#L189) | [`MOTION_ESTIMATORS`](../../src/rikka/pdr/lib/fusion/protocol.py#L117) | adaptive標準、robust中央値悪化 |
| smoothing | causal | offline | [`SMOOTHING_MODE`](../../src/rikka/common/config/__init__.py#L190) | adaptive/robust | causalは最大値重視で採用 |
| map matching | PFコマンド時のみ | 通常PDR | `use_particle_filter` | cli.commands | PF 6計測評価あり |
| 粒子数 | 500 | `--pf-particles` | [`PF_NUM_PARTICLES`](../../src/rikka/common/config/__init__.py#L167) | particle settings | 1000は最大RMSE悪化 |
| PF seed | None | 任意int | CLI/API | runner | 評価は6seed |
| 初期heading sigma | 0.03rad | 低水準API | [`PF_SIGMA_INIT_HEADING`](../../src/rikka/common/config/__init__.py#L168) | initialize | 現行採用値 |
| 歩heading sigma | 0.01rad | 低水準API | [`PF_SIGMA_HEADING`](../../src/rikka/common/config/__init__.py#L169) | propose | 現行採用値 |
| drift保持 | 0.85 | 低水準API | [`PF_HEADING_DRIFT_RETENTION`](../../src/rikka/common/config/__init__.py#L178) | propose | 長期保持は不採用 |
| stride事前 | 平均1.03、範囲0.90〜1.15 | 低水準API | `PF_STRIDE_SCALE_*` | setup/propose | 1.08以上は不採用 |
| ESS | 0.5N | 低水準API | [`PF_RESAMPLE_ESS_RATIO`](../../src/rikka/common/config/__init__.py#L179) | evaluate_map | 標準使用 |
| recovery開始 | 有効率5%未満 | 低水準API | [`PF_RECOVERY_VALID_RATIO`](../../src/rikka/common/config/__init__.py#L183) | propose | 標準使用 |
| motion重み指数 | 0.1 | `--motion-predictive-weight-power` | config | weighting | EXP-020改善 |
| 代表経路 | sequence guard | current | [`PF_PATH_SELECTION`](../../src/rikka/common/config/__init__.py#L192) | finalize | sequenceは反転改善時のみ採用 |
| BLE | 既定無効、閾値-70dBm | `--ble-landmark` | `BLE_LANDMARK_ENABLED` | BLE pipeline | 実測起点の診断を先に行う |
| BLEピーク平滑化 | 移動中央値5サンプル | Python API | `BLE_RSSI_SMOOTHING_SAMPLES` | detection | 生RSSIは診断に保持 |
| PDR BLE補正 | snap | warp / similarity | `BLE_PDR_CORRECTION_MODE` / `--ble-correction` | landmark_correction | similarityは方位・歩幅も更新 |
| similarity前向き反映 | hold | freeze | `BLE_RETROFIT_FORWARD_MODE` / `--ble-retrofit-forward` | landmark_correction | 定数方位オフセットを仮定 |
| similarity地図検査 | warn | off / enforce | `BLE_RETROFIT_MAP_CHECK` / `--ble-retrofit-map-check` | landmark retrofit | enforceは減衰後も違反なら棄却 |
| PF BLE反映 | hybrid | none / observation / reset / ranging | `PF_LANDMARK_MODE` / `--pf-landmark-mode` | particle | 実測ではranging、resetなし |
| PFアンカー経路補正 | 無効 | `--pf-landmark-retrofit` | `PF_LANDMARK_RETROFIT` | particle finalize | 代表経路だけを後処理 |
| branch quota | false | 低水準APIのみ | runner引数 | recovery | EXP-015悪化、既定無効 |
| 通常図 | 有効 | `--no-plot` | OutputSettings | plot.pipeline | 精度非影響 |
| PF animation | plot有効時 | `--save-animation` | OutputSettings | animation | 精度非影響 |
| stage/path図 | 無効 | 保存フラグ | OutputSettings | frames | EXP-028で数値一致 |

## 標準値の解決順

1. [`common.config`](../../src/rikka/common/config/__init__.py) が定数を定義します。
2. [`cli.options`](../../src/rikka/cli/options.py) がimport時にCLI既定値へコピーします。
3. 起点・方位・身長はCLI明示値、計測ディレクトリの`walk_config.csv`、共通既定値の順で解決します。
4. [`cli.commands.run()`](../../src/rikka/cli/commands.py#L81) がsettings dataclassへ変換・再検証します。
5. PDR/PF pipelineは検証済みsettingsを受けます。

設定定数を実行中にmonkeypatchしても、すでにimport済みCLIの `_..._DEFAULT` へ反映されない
可能性があります。実験ではCLI引数またはPython API引数を明示してください。
