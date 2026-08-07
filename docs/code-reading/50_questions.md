# コードから判断できなかった内容・チーム確認事項

| 分類 | 質問 | 関連ファイル・関数 | 確認できた事実 |
|---|---|---|---|
| READMEとコード不一致 | 現在の正式な既定入力はhirotoかnatsuki data5か | [`README.md`](../../README.md), [`common/config/__init__.py::DATA_DIR`](../../src/rikka/common/config/__init__.py#L21) | コードはhiroto、READMEはnatsuki data5 |
| 実験結果不足 | hiroto既定入力に対応するtruthと最新精度はあるか | [`DATA_DIR`](../../src/rikka/common/config/__init__.py#L21), `agent/agent_evaluate_*` | ログ主評価はnatsuki反復計測 |
| コメントとコード不一致 | CLI module docstringの`analyze.*`記述をいつ更新するか | [`cli/options.py`](../../src/rikka/cli/options.py) 先頭docstring | 実コードは[`common.lib.sensors`](../../src/rikka/common/lib/sensors.py)と[`cli.commands`](../../src/rikka/cli/commands.py)を遅延import |
| READMEとコード不一致 | 正解経路READMEの`src/rikka/config.py`参照を現パスへ更新するか | `input/correct_path/.../README.md` | `src/rikka/config.py`は削除済み |
| 変更履歴不一致 | `structure_before_after.md`のanalyze shim記述を履歴として残すか | [`agent/structure_before_after.md`](../../agent/structure_before_after.md) | 現在は`analyze/`なし |
| 実験ログ不一致 | EXP-004の「prewalk_robust標準」は履歴注記を追加すべきか | [`agent/EXPERIMENT_LOG.md`](../../agent/EXPERIMENT_LOG.md) EXP-004/026 | 現標準はprewalk_guarded |
| 仕様不明 | ジャイロx軸をyawとする装着姿勢の正式仕様は何か | [`common/lib/sensors.py::process_sensor_data`](../../src/rikka/common/lib/sensors.py#L71) | `df_gyro["x"]`を積分 |
| 仕様不明 | [`pixel_y_sign`](../../src/rikka/common/lib/floormap.py#L16)を重力主成分から決める対象端末姿勢は何種類か | [`common/lib/floormap.py`](../../src/rikka/common/lib/floormap.py) | gx/gzの符号で±1 |
| しきい値根拠不明 | peak 1.0m/s²、50sampleはどの端末・歩行速度で校正したか | [`step_detection.py`](../../src/rikka/pdr/lib/step_detection.py), config | 現行定数のみ確認 |
| しきい値根拠不明 | 通路gray閾値128は現map専用か | [`map_constraints.py::_is_walkable_cell`](../../src/rikka/particle/lib/map_constraints.py#L47) | 画像輝度だけで判定 |
| 複数実装の使い分け | paper step detectionとforward歩幅を使う推奨条件は何か | [`step_detection.py`](../../src/rikka/pdr/lib/step_detection.py), [`step_length.py`](../../src/rikka/pdr/lib/step_length.py) | CLI切替可能、比較表なし |
| 実験結果不足 | heading 4方式の現行同一条件比較を行うか | [`heading/resolver.py`](../../src/rikka/pdr/lib/heading/resolver.py) | 切替可能、現行横断表なし |
| 実験結果不足 | clustered/isolated/noneの最新比較を行うか | [`motion_state/clustering.py`](../../src/rikka/pdr/lib/motion_state/clustering.py) | 標準はclustered |
| 設定理由不明 | `TURNING_LENGTH_SCALE=0.3` の校正根拠は何か | [`step_motion.py`](../../src/rikka/pdr/lib/motion_state/step_motion.py), config | 数値は使用中、単独評価なし |
| 設定理由不明 | adaptive遷移行列・sigmaの学習/手調整根拠は何か | [`fusion/adaptive.py`](../../src/rikka/pdr/lib/fusion/adaptive.py) | 定数はコード内固定 |
| 用途不明 | [`_snap_trajectory_to_walkable_pixels()`](../../src/rikka/particle/lib/map_constraints.py#L170)を今後使う予定か | [`particle/lib/map_constraints.py`](../../src/rikka/particle/lib/map_constraints.py) | 現行呼び出し元なし |
| 評価仕様 | recovery後ESS=Nを品質指標としてどう解釈するか | [`evaluate_map.py`](../../src/rikka/particle/lib/evaluate_map.py), EXP-027 | 一様化で最大になる |
| 仕様不明 | sensor取得側のsampling rate・軸向き・端末固定方法はどこで管理するか | リポジトリ外 | CSV以降のみ実装 |
| 精度課題 | recovery driftが残す25〜90°曲がりを合否指標へ追加するか | EXP-027, [`sequence_path.py`](../../src/rikka/particle/lib/sequence_path.py) | 現行反転判定は135°以上 |
| 削除可否 | 古いreference/agent計画書の現パス不一致を履歴として許容するか | `reference/*.md`, `agent/PF_*PLAN.md` | 多数が旧`analyze/config.py`を参照 |

## 会議で先に決めたい3点

1. 正式な既定入力と、その正解・受入基準。
2. sensor取得時の軸・装着・sampling仕様をどこへ記録するか。
3. recovery後の滑らかだが誤った曲がりを検出する追加指標を採用するか。
