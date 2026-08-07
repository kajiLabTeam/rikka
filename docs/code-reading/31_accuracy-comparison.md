# 精度比較

## 表で使う記号

[`agent/EXPERIMENT_LOG.md`](../../agent/EXPERIMENT_LOG.md) の表記に合わせています。

- 「無印」= [`input/sensor_data/natsuki/1turn_rightsidestep_3turn_leftsidestep`](../../input/sensor_data/natsuki/1turn_rightsidestep_3turn_leftsidestep)（末尾に数字なし）
- 「5〜9」= 同じルートの反復計測 `...leftsidestep5` 〜 `...leftsidestep9`
- 正解軌跡は [`input/correct_path/1turn_rightsidestep_3turn_leftsidestep/`](../../input/correct_path/1turn_rightsidestep_3turn_leftsidestep) を共通で使用

## 同一条件で比較できる結果

| 比較 | 評価データ・条件 | 指標 | 変更前 | 変更後 | 判断・情報源 |
|---|---|---|---:|---:|---|
| legacy→adaptive PDR | 無印+5〜8、同じtruth | 各RMSE | 2.656/6.865/7.963/4.346/3.818 | 2.166/6.728/7.838/3.364/3.622m | 全5改善、EXP-018 |
| PF motion power | 5記録×6seed、adaptive causal | 全体RMSE中央値/最大 | 4.933/11.231 | 4.052/9.727m | power 0.1採用、EXP-020 |
| robust PDR | 同じ5記録 | RMSE中央値/最大 | adaptive 3.622/7.838 | robust 5.352/7.390m | 最大改善・中央値悪化、EXP-021 |
| 横歩き倍率 | 同じ通常PDR 5記録 | 各RMSE | 倍率1.0: 2.804/7.549/8.229/5.186/4.202 | 0.8: 2.656/6.865/7.963/4.346/3.818m | 0.8暫定採用、EXP-016 |
| 到達可能クラスタ | 無印PF 6seed | RMSE中央値/最大 | 1.221/2.427 | 1.428/2.452m | 安全性維持、形状小幅悪化、EXP-029 |
| gyro/PF最終構成 | 無印+5〜9、PF6seed | PDR RMSE中央値/最大 | 5.175/8.149 | 3.313/5.183m | 複合変更、EXP-026 |
| gyro/PF最終構成 | 同上36 PF実行 | PF RMSE中央値/最大 | 4.907/13.291 | 2.064/6.554m | 壁/recovery/終端failure 0、EXP-026 |

「最終構成」はguard付きbiasと記録品質適応PFノイズ等を含むため、単一パラメータの
因果効果としては扱いません。

## 条件が異なるため参考値のみ

| 結果 | 条件 | 参考値 | 比較上の注意 |
|---|---|---:|---|
| 現行構造整理ベースライン | 無印、6seed、EXP-030 | PF 1.428/2.452m | 6計測横断値とは別母集団 |
| 過去標準データ | 無印、6seed、EXP-022 | PF 0.858/2.536m | 代表経路実装などが現行と異なる |
| 6計測最終 | 無印+5〜9、36実行、EXP-026 | PF 2.064/6.554m | データ数・集計が異なる |
| performance | 既定500粒子95歩seed0 | PF内部約7.6s、finalize約5.0s | cProfile環境依存、精度指標ではない |

## 評価結果が存在しない、または同一条件比較不可

| 処理 | 状態 |
|---|---|
| `peak` vs `paper_vertical_threshold` | 歩数誤差・RMSEの同一条件表を確認できない |
| Weinberg vs forward歩幅 | 現行同一条件比較を確認できない |
| heading `gyro/accel_method1/accel_method2/gyro_accel_motion` | 方式だけを変えた現行横断表なし |
| forward heading body vs motion | 過去診断はあるが、現行6計測の確定表なし |
| sidestep heading motion/body_lateral/blend | 同一条件表なし |
| `clustered/isolated/none` | 一部実験記述はあるが現行統一表なし |
| 現在の既定hiroto入力 | 対応truthと最新数値評価を確認できない |

## ログ上は成功しているが数値評価がない処理

- `paper_vertical_threshold` は回帰テスト・CSV出力経路がありますが、歩数正解との数値表はありません。
- MP4→GIF fallback、CSV/図保存は動作確認対象であり精度比較対象ではありません。
- Phase G import移行はgolden一致・180 tests成功ですが、アルゴリズム精度改善ではありません。

## 比較時の注意

- truthとセンサーは非同期のため、正規化弧長比較です。
- 同じ数値でも、対象記録数、seed、代表経路実装が違えば単純比較しません。
- 壁交差0は形状精度を保証しません。
- RMSE中央値だけでなく最大値、終端方向、recovery failureも確認します。
