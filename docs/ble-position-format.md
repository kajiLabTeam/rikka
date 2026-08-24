# BLE 端末座標の記録形式

BLE 端末の既知座標は、各計測データのディレクトリに置く `BLE_pos.csv` を
正本として記録する。例えば `WithBLE_1` の座標ファイルは
`input/senser_data_withBLE/natsuki/WithBLE_1/BLE_pos.csv` とする。

同じディレクトリに `Accelerometer.csv`、`Gyroscope.csv`、`BLE.csv`、
`Time Reference.csv`、phyphoxが出力する `meta/time.csv` を置く。次のように
BLE補正を有効にすると、`-d` のディレクトリにある `BLE.csv` と
`BLE_pos.csv` が自動的に選択される。

```sh
uv run rikka run \
  -d input/senser_data_withBLE/natsuki/WithBLE_1 \
  --ble-landmark \
  --no-plot
```

`BLE.csv` の `Timestamp` は日本時間として解釈し、`meta/time.csv` の
`START` 行にある `system time - experiment time` を引いて、phyphoxの
`Experiment Time (s)` と同じ時間軸へ変換する。`meta/time.csv` がない旧データ
だけは `Time Reference.csv` の `Unix Offset (s)` を使用する。`RSSI=127` は
無効値として除外する。

## 座標系

- 対象画像: `input/Floormap_building14_5floor.png`
- 原点: 画像左上 `(0, 0)`
- X 軸: 右向きが正
- Y 軸: 下向きが正
- 単位: pixel
- 有効範囲: `0 <= pixel_x < 2837`、`0 <= pixel_y < 3742`

`pixel_x` と `pixel_y` には、端末を実際に置いた位置の画像上の座標を
単位なしの数値で入力する。位置をまだ確認していない端末は空欄のままにし、
推測値を入れない。

## CSV の列

| 列 | 必須 | 内容 |
|---|---|---|
| `beacon_id` | 必須 | Rikka 内で使う一意な識別子。現在は Device Name と同じ値を使う |
| `device_name` | 必須 | BLE ログの `Device Name` |
| `mac_address` | 必須 | BLE ログの `MAC Address` |
| `raw_data_suffix` | 必須 | 広告形式が変わっても共通する `Raw Data` 末尾の16進文字列 |
| `pixel_x` | 座標確定後に必須 | フロアマップ上の X 座標 [pixel] |
| `pixel_y` | 座標確定後に必須 | フロアマップ上の Y 座標 [pixel] |
| `note` | 任意 | 設置場所を人が確認するための短い説明 |

ファイルは UTF-8 の CSV とし、ヘッダー名は変更しない。1端末につき1行にする。
BLE の時刻や RSSI は `BLE.csv` に残し、`BLE_pos.csv` には入れない。

入力例:

```csv
beacon_id,device_name,mac_address,raw_data_suffix,pixel_x,pixel_y,note
elpis_001,elpis_001,DC:0D:30:1E:33:91,656c7069735f303031,2056,2400,廊下中央
```

プログラムでは `beacon_id` を座標との結合キーにする。
端末照合には Device Name に加えて MAC Address と Raw Data の末尾も使用できる。
