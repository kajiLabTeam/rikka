# PF既定方式の切り替え（2026-10-22）

ユーザー指定により、PF既定をcandidate-bank K=8へ変更した。
歩検出は既定のadaptive_peakを維持する。

```sh
# 新方式（追加指定は不要）
uv run rikka particle

# 従来のPF（歩検出はadaptive_peakのまま）
uv run rikka particle --pf-proposal legacy

# 歩検出も従来方式にする場合
uv run rikka particle --pf-proposal legacy --step-detection peak
```

`PF_PROPOSAL_METHOD` と `PF_CANDIDATE_COUNT` を共通設定へ追加し、CLI・
`cli.commands.run()`・`ParticleSettings` の既定を合わせた。
候補数の省略時はcandidate-bankで8、legacy/no-relocationで1へ解決する。
明示した不正な候補数は引き続き拒否する。
低水準の `run_particle_steps()` の従来互換既定は維持した。

候補生成の数式・重み・再標本化・復旧・経路選択は変更していない。
フェーズ3・4の実装や、フェーズ2の測位精度改善を確認したという判断ではない。
実BLE観測を用いるcandidate-bankの制限も維持するため、その場合はlegacyを明示する。

## 確認

- `rikka particle --help`: candidate-bank、候補数はcandidate-bankで8/その他1と表示。
- CLI回帰: オプションなしでbank8、legacy/no-relocationを指定したときは候補数を追加せず1へ解決することを確認。
- 関連テスト211件成功、Ruff・Mypy・差分の空白検査成功。
- 全体テストは503件成功・8件失敗。失敗したテスト名は前回の全体評価と同じ8件（BLE地図/読込2、landmark4、従来PF golden2）。全体成功とは報告しない。
- 実行データは1021_計測1回目、seed 0、`--no-plot`。PF方式を指定しない実行が、保存した明示bank8の経路とCSV丸め許容差2e-14m以内で一致した（88歩、最終経路長79.968m）。
- `--pf-proposal legacy` の実行も、保存した明示legacyの経路と同じ許容差で一致した。
- 比較元は [48実行の再評価](pf-phase2-adaptive-reevaluation-20261022/README.md)。既定の選択のみの変更なので、その数値比較を再実行して採用判断を更新してはいない。
- 生成した `output/20261022_172713_098108` と `output/20261022_172831_035718` だけを確認後削除。既存output一覧の不変を確認した。入力・既存実験ログは変更していない。
- 既存BLEテストは対応済みlegacyを、従来goldenは固定peak＋legacyを明示し、既定選択のテストと分離した。期待値やgoldenデータは書き換えていない。

検証結果の数値は `pf-default-bank8-validation-20261022.json` に保存した。
