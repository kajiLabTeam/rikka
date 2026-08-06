# 主要データ構造

## DataFrame

| データ | 入力元・生成元 | 型・単位 | 主な列 | 主な利用先・注意点 |
|---|---|---|---|---|
| 生加速度 | `Accelerometer.csv` → [`load_sensor_data()`](../../src/rikka/common/lib/sensors.py#L51) | `DataFrame`, s / m/s² | `t,x,y,z` | `t` は入力にあれば使用。XYZは端末座標 |
| 前処理加速度 | [`process_sensor_data()`](../../src/rikka/common/lib/sensors.py#L71) | `DataFrame`, m/s² | `gx,gy,gz`, `lin_*`, `lin_norm`, `low_lin_norm`, `v_acc`, `h_*`, `h_norm` | `v_acc` は重力方向射影、`h_*` は重力成分除去後 |
| 生ジャイロ | `Gyroscope.csv` → `load_sensor_data()` | `DataFrame`, s / rad/s | `t,x,y,z` | 現行yaw積分は `x` 軸を使用 |
| 前処理ジャイロ | `process_sensor_data()` | `DataFrame`, rad/s / rad | `gyro_rate`, `gyro_bias`, `angle`, `low_angle` | `attrs["gyro_bias_result"]` に診断型を保持 |
| 軌跡CSV | [`plot.lib.outputs`](../../src/rikka/plot/lib/outputs.py) | `DataFrame`, s / m | `timestamp_s,x,y` | 原点 `[0,0]` は含まず1行=1歩。先頭行は1歩目の到達位置で、`timestamp_s` は1歩目を0とした相対時刻 |

## 共有型

| 型 | 生成 | 主なフィールド | 利用 |
|---|---|---|---|
| [`StepDetectionResult`](../../src/rikka/common/lib/models.py#L39) | [`detect_step_result()`](../../src/rikka/pdr/lib/step_detection.py#L166) | `method`, `peaks`, `segments`, `threshold`, `polarity` | 方位・歩幅・出力 |
| [`StepHeading`](../../src/rikka/common/lib/models.py#L49) | [`resolve_step_heading()`](../../src/rikka/pdr/lib/heading/resolver.py#L109) 以降 | body/motion/selected heading、移動分類、変位特徴、信頼度 | PDR、PF、CSV、図 |
| [`StepLengthObservation`](../../src/rikka/common/lib/models.py#L158) | [`build_step_length_observation()`](../../src/rikka/pdr/lib/step_length.py#L79) | nominal/interval length、周期、振幅、quality、sigma | adaptive PDR |
| [`StepMotionEvidence`](../../src/rikka/common/lib/models.py#L123) | [`build_step_motion_evidences()`](../../src/rikka/pdr/lib/motion_state/evidence.py#L123) | 4状態尤度、motion/calibration reliability | adaptive PDR、PF |
| [`StepMotionPosterior`](../../src/rikka/common/lib/models.py#L172) | [`estimate_adaptive_pdr()`](../../src/rikka/pdr/lib/fusion/adaptive.py#L288) | 4状態確率、方位・歩幅・offsetの平均/標準偏差 | PDR確定値、PF proposal |
| [`PreparedPdrSteps`](../../src/rikka/common/lib/models.py#L230) | [`prepare_pdr_steps_with_settings()`](../../src/rikka/pdr/lib/preparation.py#L111) | 前処理DF、歩列、通常軌跡、観測・事後分布 | PDR/PF境界 |
| [`ParticleFilterResult`](../../src/rikka/common/lib/models.py#L260) | [`particle.pipeline.run_particle()`](../../src/rikka/particle/pipeline.py#L28) | 代表軌跡、各歩の粒子、診断、可視化stage | 出力領域 |
| [`TrajectoryResult`](../../src/rikka/common/lib/models.py#L280) | PDR/PF pipeline | 軌跡、歩幅、時刻、方位、prepared、任意particle | CLI/plot境界 |

## StepHeading の重要フィールド

| フィールド | 単位・値 | 意味 |
|---|---|---|
| `gyro_heading` / `body_heading` | rad, `[-π,π)` | ジャイロ積分を起点とする端末・身体方位候補 |
| `motion_heading` | rad, `[-π,π)` | 水平加速度の二重積分から得た世界座標移動方位 |
| `selected_heading` | rad, `[-π,π)` | 最終的に軌跡/PFへ渡す方位 |
| `forward_displacement` | m相当の積分特徴 | body軸方向の符号付き変位特徴 |
| `lateral_displacement` | m相当の積分特徴 | body左方向を正とする横変位特徴 |
| `movement_type` | 文字列 | センサー観測上の分類 |
| `trajectory_movement_type` | 文字列またはNone | 平滑化・fusion後に軌跡で使う分類 |
| `step_length_scale` | 比率 | 横歩き0.8、旋回0.3などの適用倍率 |
| `yaw_delta` | rad | 1歩区間内のジャイロ角変化 |

すべての heading は内部では rad、CSV出力時に度列へ変換されます。

## 粒子状態

[`ParticleRuntime`](../../src/rikka/particle/lib/state.py#L36) が shape `(N,2)` の `particles` [m]、正規化 `weights`、
`heading_drift` [rad]、`stride_scale`、4値 `motion_state`、親インデックスと履歴を
保持します。開始位置は全粒子 `[0,0]` です。初期ばらつきは方位 `heading_drift`
（標準偏差0.03 rad）と歩幅倍率 `stride_scale`（平均1.03・標準偏差0.05を0.90〜1.15へclip）
の2つに与えます。

## マップと座標系

| 項目 | 内容 |
|---|---|
| 推定座標 | 右手系2D、`x += length*cos(theta)`, `y += length*sin(theta)`、m |
| 方位 | 0°=+X、90°=+Y、反時計回り正 |
| 画像座標 | x右向き、y下向き、pixel |
| 原点 | `FLOORMAP_ORIGIN_PX=(2050,400)` |
| 縮尺 | `0.01 m/pixel` |
| Y符号 | `gx_mean/gz_mean` から [`pixel_y_sign()`](../../src/rikka/common/lib/floormap.py#L16) が ±1 を決定 |
| 通路判定 | 正規化グレースケール `>128` |

## 評価結果

評価スクリプトは専用 dataclass ではなく辞書・JSON/CSVを中心に扱います。主な値は
`arc_rmse_m`、`endpoint_error_m`、推定/正解距離、終端方向指標、壁交差数、
recovery failure数です。時刻同期ではなく、開始点を合わせた正規化弧長300点で比較します。
