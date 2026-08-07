# 画面共有コードリードガイド

## 推奨進行

「必ず読む」11本で約90分、「時間があれば」以降も含めると約110分です。

| 順番 | 優先度 | 目安 | Markdown | 開くコード | 確認する関数・データ | 議論するポイント |
|---:|---|---:|---|---|---|---|
| 1 | 必ず読む | 5分 | [`01_system-flow.md`](01_system-flow.md) | [`src/rikka/cli/commands.py`](../../src/rikka/cli/commands.py) | `run`, [`TrajectoryResult`](../../src/rikka/common/lib/models.py#L280) | 推定・PF・plotの境界 |
| 2 | 必ず読む | 8分 | [`02_entry-and-execution.md`](02_entry-and-execution.md) | [`src/rikka/__init__.py`](../../src/rikka/__init__.py), [`cli/options.py`](../../src/rikka/cli/options.py) | `main`, [`_common_options`](../../src/rikka/cli/options.py#L172) | CLI既定の伝播 |
| 3 | 必ず読む | 8分 | [`03_data-structures.md`](03_data-structures.md) | [`common/lib/models.py`](../../src/rikka/common/lib/models.py) | [`StepHeading`](../../src/rikka/common/lib/models.py#L49), [`PreparedPdrSteps`](../../src/rikka/common/lib/models.py#L230) | body/motion/selectedの違い |
| 4 | 必ず読む | 8分 | [`10_data-loading-and-preprocessing.md`](10_data-loading-and-preprocessing.md) | [`common/lib/sensors.py`](../../src/rikka/common/lib/sensors.py) | [`process_sensor_data`](../../src/rikka/common/lib/sensors.py#L71) | 重力分離、bias、実時刻 |
| 5 | 必ず読む | 7分 | [`11_step-detection.md`](11_step-detection.md) | [`pdr/lib/step_detection.py`](../../src/rikka/pdr/lib/step_detection.py) | [`detect_step_result`](../../src/rikka/pdr/lib/step_detection.py#L166) | peakとpaperのしきい値 |
| 6 | 必ず読む | 12分 | [`12_heading-estimation.md`](12_heading-estimation.md) | [`pdr/lib/heading/resolver.py`](../../src/rikka/pdr/lib/heading/resolver.py), [`heading/motion.py`](../../src/rikka/pdr/lib/heading/motion.py) | [`resolve_step_heading`](../../src/rikka/pdr/lib/heading/resolver.py#L109) | 端末方位と移動方位 |
| 7 | 必ず読む | 12分 | [`13_motion-state-estimation.md`](13_motion-state-estimation.md) | [`motion_state/refinement.py`](../../src/rikka/pdr/lib/motion_state/refinement.py), [`motion_state/clustering.py`](../../src/rikka/pdr/lib/motion_state/clustering.py), [`fusion/protocol.py`](../../src/rikka/pdr/lib/fusion/protocol.py), [`fusion/adaptive.py`](../../src/rikka/pdr/lib/fusion/adaptive.py) | [`refine_step_headings_with_motion_model`](../../src/rikka/pdr/lib/motion_state/refinement.py#L170)、cluster、[`MOTION_ESTIMATORS`](../../src/rikka/pdr/lib/fusion/protocol.py#L117) | 横歩き確定とadaptive |
| 8 | 必ず読む | 7分 | [`14_step-length-and-trajectory.md`](14_step-length-and-trajectory.md) | [`step_length.py`](../../src/rikka/pdr/lib/step_length.py), [`integrate.py`](../../src/rikka/pdr/lib/integrate.py) | Weinberg、[`integrate_steps`](../../src/rikka/pdr/lib/integrate.py#L18) | 歩幅scaleの順序 |
| 9 | 必ず読む | 12分 | [`15_particle-filter.md`](15_particle-filter.md) | [`particle/lib/runner.py`](../../src/rikka/particle/lib/runner.py), [`propose.py`](../../src/rikka/particle/lib/propose.py) | 1歩loop、粒子状態 | proposalとweight |
| 10 | 必ず読む | 8分 | [`16_map-matching-and-recovery.md`](16_map-matching-and-recovery.md) | [`evaluate_map.py`](../../src/rikka/particle/lib/evaluate_map.py), [`map_constraints.py`](../../src/rikka/particle/lib/map_constraints.py) | [`resolve_map_constraints`](../../src/rikka/particle/lib/evaluate_map.py#L98) | 壁判定と復旧順 |
| 11 | 必ず読む | 5分 | [`19_output-and-visualization.md`](19_output-and-visualization.md) | [`plot/pipeline.py`](../../src/rikka/plot/pipeline.py) | [`write_outputs`](../../src/rikka/plot/pipeline.py#L51), `render` | 保存との接続 |
| 12 | 時間があれば読む | 8分 | [`18_evaluation.md`](18_evaluation.md) | [`agent_benchmark_pdr_pf_methods.py`](../../agent/agent_benchmark_pdr_pf_methods.py) | [`_trajectory_metrics`](../../agent/agent_benchmark_pdr_pf_methods.py#L198) | 弧長評価の前提 |
| 13 | 時間があれば読む | 5分 | [`30_active-configurations.md`](30_active-configurations.md) | [`common/config/__init__.py`](../../src/rikka/common/config/__init__.py) | 標準値 | 変更理由・校正範囲 |
| 14 | 今回は概要のみ | 5分 | [`31_accuracy-comparison.md`](31_accuracy-comparison.md) | [`agent/EXPERIMENT_LOG.md`](../../agent/EXPERIMENT_LOG.md) | EXP-020/026/029 | 条件差を混ぜない |

## 最短45分コース

順番1（5分）、3（8分）、6（12分）、7（12分）、11（5分）の計42分で、全体像・データ構造・
方位・運動状態・保存だけを押さえます。
PF が主題の会議では、順番6または7を順番9（[`15_particle-filter.md`](15_particle-filter.md)、12分）と入れ替えます。
復旧処理まで踏み込む場合は順番10（8分）を追加し、50分程度を見込んでください。

## 最初に開くべきコード

[`src/rikka/cli/commands.py::run()`](../../src/rikka/cli/commands.py#L81) です。ここから通常PDR、任意PF、保存の3境界が一度に
見えます。次に [`pdr/lib/preparation.py::prepare_pdr_steps_with_settings()`](../../src/rikka/pdr/lib/preparation.py#L111) を開くと、
内部推定の処理順を追えます。

## 会議中に手元に置くデータ

- `StepHeading` の主要フィールド表
- `step_headings.csv` の1〜3行
- `particle_diagnostics.csv` の通常歩・recovery歩を各1行
- [`30_active-configurations.md`](30_active-configurations.md)
- [`50_questions.md`](50_questions.md)
