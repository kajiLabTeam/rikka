# rikka

スマートフォンの加速度計・ジャイロスコープ CSV から歩行軌跡を推定する
PDR（Pedestrian Dead Reckoning）ライブラリです。

現在の標準設定は、ジャイロを体の向き、水平加速度を横歩き判定用の
移動特徴として使う `gyro_accel_motion` です。通常歩行の軌跡方位は
`body_heading` を使い、横歩き判定は記録として残しつつ、同方向にまとまった
横歩きだけを軌跡へ反映します。

## セットアップ

```sh
uv sync --all-groups
```

AI エージェントや sandbox 環境で `uv` のホームキャッシュ権限に引っかかる場合は、
リポジトリ内キャッシュを使います。

```sh
UV_CACHE_DIR=.uv-cache uv run pytest
```

パーティクルフィルタの MP4 アニメーション出力には `ffmpeg` が必要です。

```sh
brew install ffmpeg
```

## 入力データ

`input/<データフォルダ>/` に phyphox 形式の CSV を置きます。

```text
input/
└── my_walk/
    ├── Accelerometer.csv
    └── Gyroscope.csv
```

対応する列名は次の通りです。

| ファイル | 対応列 |
|---|---|
| `Accelerometer.csv` | `Time (s)`, `Acceleration x/y/z (m/s^2)` または `X/Y/Z (m/s^2)` |
| `Gyroscope.csv` | `Time (s)`, `Gyroscope x/y/z (rad/s)` または `X/Y/Z (rad/s)` |

現在の既定入力は次です。

```python
DATA_DIR = "input/1turn_rightsidestep_3turn_leftsidestep4"
```

別データを使う場合は、CLI の `-d` で指定できます。

```sh
UV_CACHE_DIR=.uv-cache uv run rikka run -d input/my_walk
```

## 基本コマンド

標準設定で通常 PDR を実行します。プロット表示と PNG/CSV 保存を行います。

```sh
UV_CACHE_DIR=.uv-cache uv run rikka run
```

プロット表示を止めてバッチ確認する場合だけ `--no-plot` を付けます。

```sh
UV_CACHE_DIR=.uv-cache uv run rikka run --no-plot
```

`pdr` は `run` の別名です。

```sh
UV_CACHE_DIR=.uv-cache uv run rikka pdr
```

パーティクルフィルタ付きで実行します。

```sh
UV_CACHE_DIR=.uv-cache uv run rikka particle
```

グラフ表示なしでパーティクルフィルタのアニメーションだけ保存する場合:

```sh
UV_CACHE_DIR=.uv-cache uv run rikka particle --no-plot --save-animation
```

センサー波形を確認します。入力フォルダに `sensor_plot.png` を保存します。

```sh
UV_CACHE_DIR=.uv-cache uv run rikka sensor
```

## 標準設定

`uv run rikka run` の現在の主要な既定値です。

| 項目 | 既定値 | 説明 |
|---|---:|---|
| `DATA_DIR` | `input/1turn_rightsidestep_3turn_leftsidestep4` | 入力データ |
| `FLOORMAP_PATH` | `input/Floormap_building14_5floor.png` | 背景マップ |
| `FLOORMAP_ORIGIN_PX` | `(2050, 600)` | 軌跡の開始ピクセル |
| `FLOORMAP_SCALE` | `0.01` | 1px あたりのメートル数 |
| `INITIAL_DIRECTION` | `90.0` | 歩行開始方向 [deg] |
| `STEP_DETECTION_METHOD` | `peak` | ステップ検出 |
| `HEADING_METHOD` | `gyro_accel_motion` | 方位・移動方向推定 |
| `FORWARD_HEADING_SOURCE` | `body` | 通常歩行の軌跡方位は体/端末方向を使用 |
| `GYRO_BIAS_METHOD` | `prewalk_robust` | ジャイロバイアス推定 |
| `USER_HEIGHT_M` | `1.65` | Weinberg 歩幅補正用の身長 |
| `SIDESTEP_LATERAL_RATIO` | `1.2` | 横方向/前方向の比率がこの値以上で横歩き候補 |
| `SIDESTEP_MIN_LATERAL_DISPLACEMENT_M` | `0.03` | 横歩き判定に必要な横方向変位 [m] |
| `SIDESTEP_SMOOTHING_METHOD` | `clustered` | 5歩窓で同方向横歩きが2回以上ある場合だけ軌跡へ横歩きとして反映 |
| `SIDESTEP_LENGTH_SCALE` | `1` | 横歩き歩幅の倍率 |
| `TURNING_LENGTH_SCALE` | `0.3` | 旋回中歩幅の倍率 |

`gyro_accel_motion` では次を分けて扱います。

- `body_heading`: ジャイロから推定した体/端末の向き
- `motion_heading`: 水平加速度から推定した1歩ごとの移動方向特徴
- `movement_type`: センサー上の判定結果
- `trajectory_movement_type`: 軌跡計算に使った移動タイプ
- `forward_heading_source`: `forward` 判定ステップの軌跡方位ソース

## 横歩き判定の見方

実行後、`output/<timestamp>/step_headings.csv` を確認します。

重要な列は次です。

| 列 | 意味 |
|---|---|
| `movement_type` | センサー上の判定。`sidestep_left/right` なら横歩きとして検出済み |
| `trajectory_movement_type` | 軌跡に反映した移動タイプ。`forward` なら検出は横歩きだが軌跡上は前進扱い |
| `body_heading_deg` | 体/端末の向き |
| `motion_heading_deg` | 水平加速度から見た移動方向 |
| `selected_heading_deg` | 実際に軌跡へ使った方位 |
| `forward_heading_source` | `forward` 判定ステップの軌跡方位ソース |
| `lateral_forward_ratio` | 横方向変位 / 前方向変位 |
| `forward_displacement` | 体方向への変位特徴 |
| `lateral_displacement` | 横方向への変位特徴 |
| `motion_heading_correction_deg` | 水平加速度方向の補正角 |

標準設定では、単発の横歩き判定は `movement_type` には残りますが、
`trajectory_movement_type="forward"` として軌跡には前進扱いで反映されます。
一方、5歩窓で同方向の横歩きが2回以上ある場合は、横歩き区間として軌跡へ反映されます。
また、`forward` 判定ステップは `body_heading` で軌跡へ積みます。
旧挙動のように水平加速度由来の `motion_heading` で積む場合は次を指定します。

```sh
UV_CACHE_DIR=.uv-cache uv run rikka run --forward-heading-source motion
```

横歩き判定を軌跡へそのまま反映して比較したい場合:

```sh
UV_CACHE_DIR=.uv-cache uv run rikka run --sidestep-smoothing none
```

旧方式の単発横歩き抑制と比較したい場合:

```sh
UV_CACHE_DIR=.uv-cache uv run rikka run --sidestep-smoothing isolated
```

横歩き判定を増やす/減らす場合:

```sh
# 拾いやすくする
UV_CACHE_DIR=.uv-cache uv run rikka run --sidestep-lateral-ratio 1.0

# 厳しくする
UV_CACHE_DIR=.uv-cache uv run rikka run --sidestep-lateral-ratio 1.5
```

## ジャイロバイアス比較

2つ目のサンプルデータは `prewalk_robust` のバイアス推定で軌跡が曲がりやすいです。
比較用には手動バイアスも使えます。

```sh
UV_CACHE_DIR=.uv-cache uv run rikka run \
  -d input/1turn_rightsidestep_3turn_leftsidestep2 \
  --gyro-bias-method manual \
  --gyro-bias 0.002
```

## 主な出力

通常 PDR は `output/<timestamp>/` に保存します。

| ファイル | 内容 |
|---|---|
| `trajectory.csv` | 移動後座標。列は `timestamp_s,x,y` |
| `trajectory.png` | フロアマップ上の軌跡 |
| `step_lengths.csv` | ステップごとの歩幅 |
| `step_lengths.png` | 歩幅グラフ |
| `step_vectors.csv` | ステップごとの変位ベクトル |
| `step_vectors/step_*.png` | 各ステップの変位と加速度分布 |
| `step_headings.csv` | 方位候補、横歩き判定、軌跡反映タイプ |
| `gyro_bias.csv` | ジャイロバイアス推定の診断情報 |
| `step_segments.csv` | `paper_vertical_threshold` 使用時のステップ区間 |

パーティクルフィルタでは追加で次を保存します。

| ファイル | 内容 |
|---|---|
| `pf_trajectory.png` | PF の平均軌跡 |
| `particle_filter.mp4` / `.gif` | パーティクル分布アニメーション |

## Python から使う

```python
import pandas as pd
from rikka.analyze.pdr import run

df_acc = pd.DataFrame(...)   # 列: t, x, y, z
df_gyro = pd.DataFrame(...)  # 列: t, x, y, z

trajectory = run(df_acc=df_acc, df_gyro=df_gyro)
```

`df_acc` と `df_gyro` は両方渡すか、両方省略してください。片方だけ渡すと
`ValueError` になります。

グラフを表示しない場合:

```python
trajectory = run(df_acc=df_acc, df_gyro=df_gyro, plot=False)
```

パーティクルフィルタを使う場合:

```python
trajectory = run(
    df_acc=df_acc,
    df_gyro=df_gyro,
    use_particle_filter=True,
)
```

## 開発コマンド

```sh
UV_CACHE_DIR=.uv-cache uv run ruff format
UV_CACHE_DIR=.uv-cache uv run ruff check
UV_CACHE_DIR=.uv-cache uv run mypy src/
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv build
```

CI と同等の pre-commit チェック:

```sh
UV_CACHE_DIR=.uv-cache uv run pre-commit run --all-files
```

リポジトリ管理の Git hook を使う場合:

```sh
git config core.hooksPath .githooks
```

## 接続確認

```python
from rikka import ping

print(ping())  # Hello, rikka
```

![sensor_plot サンプル](docs/images/sensor_plot_sample.png)
![step_lengths サンプル](docs/images/step_lengths_sample.png)
