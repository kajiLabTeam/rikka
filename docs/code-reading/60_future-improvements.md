# 今後の改善案

今回は提案のみです。実装、ファイル移動、しきい値変更は行っていません。

| 分類 | 現状 | 問題 | 改善案 | 影響範囲 | 優先度 | 変更時のリスク |
|---|---|---|---|---|---|---|
| ドキュメント改善 | READMEのDATA_DIRがコードと不一致 | 実行対象を誤認 | 既定入力を一元引用またはCI照合 | README/config | 高 | 意図確認なしに一方へ揃えない |
| ドキュメント改善 | 古いagent/referenceが旧パスを記載 | 現コードへ移動しにくい | 各資料へ「履歴・現パス」注記 | agent/reference | 中 | 履歴事実を消さない |
| 設定管理 | CLIがimport時に定数をコピー | runtime monkeypatchと不一致 | settings factoryからCLI既定を生成 | config/options | 中 | help・公開API既定の互換性 |
| 設定管理 | adaptive/PF低水準値がconfigと関数引数へ広く展開 | 実験条件記録が長い | PDR/PF設定の階層型を低水準runnerまで渡す | settings/particle | 中 | 固定seedの乱数・数値順序 |
| 型定義 | `ParticleFilterResult`の診断が`tuple[Any,...]` | 境界型が弱い | recorder型を循環なしで共有型化 | models/recorder | 低 | import循環 |
| 関数分割 | `run_particle_steps`は42引数 | 読解・呼出ミス | prepared/config/runtime factoryへ束ねる | particle API/tests | 中 | 外部低水準API互換 |
| 責務整理 | `cli.commands.run()`が実行と保存を常に結合 | 純粋APIでもoutputを生成 | analyze結果返却と保存adapterを分ける | CLI/Python API | 中 | 現公開APIの副作用変更 |
| 命名改善 | `body_heading`が端末/身体候補を兼ねる箇所 | 意味が段階で変わる | device/body estimateを型で分離 | heading/models/CSV | 中 | CSV/API互換 |
| ログ改善 | recovery後ESSがN | 品質良好と誤読 | ESSに加えpre-recovery massとmodeを主表示 | diagnostics/plot | 高 | 診断列順・golden |
| テスト追加 | 現既定入力のtruth評価なし | README/標準の精度を保証できない | 正式truth確定後に複数seed gate追加 | agent/tests/data | 高 | データ容量・過適合 |
| テスト追加 | 歩検出2方式の歩数比較なし | 切替判断不能 | 歩イベント正解ラベルを用意 | step_detection/tests | 中 | ラベル作成コスト |
| テスト追加 | Weinberg/forward比較なし | 歩幅方式の推奨条件不明 | 歩ごとの距離正解を別収集 | step_length/agent | 中 | 軌跡truthから歩幅を逆算しない |
| アルゴリズム切替 | 方式比較が複数scriptへ分散 | 条件の取り違え | manifest付き統一benchmarkを標準化 | agent/skills | 中 | 過去結果とのschema互換 |
| 未使用コード整理 | `_snap_trajectory_to_walkable_pixels`の呼出なし | 用途判断が必要 | 意図確認後、実験用明示化か削除提案 | map_constraints/tests | 低 | 隠れた外部利用 |
| ファイル分割 | `cli/options.py`等が400行超 | 目的のoption探索に時間 | コマンド別decoratorへ分割 | CLI | 低 | Click decorator順序 |
| 精度評価 | 135°未満の不自然な曲がりを合否に含めない | EXP-027を検出できない | sensor根拠付き曲率/残差指標を追加 | evaluation/PF | 高 | 実際の急旋回を誤検出 |
| センサー仕様 | 取得条件がrepo外 | 軸・Hzの再現性不足 | 入力データごとのmanifestを追加 | input/docs/loader | 高 | 個人情報を含めない設計 |

## 優先順位の考え方

最初に「正式な既定入力・truth・取得条件」を確定し、その後に評価gateを追加するのが安全です。
構造改善は固定seed CSV一致と複数seed安全性を維持できる場合だけ進めます。
