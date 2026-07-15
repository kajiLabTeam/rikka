# PDR テスト対応表

変更した領域に対応する絞り込みテストを先に実行し、その後に全回帰テストを実行する。
`-k` の条件は変更した関数名や近接するテスト名に合わせて狭めてよい。

| 変更領域 | 主な対象 | 絞り込み例 |
|---|---|---|
| センサー前処理・時刻 | `sensors.py`, `time_utils.py` | `uv run pytest tests/test_pdr_regressions.py -k "sensor or gyro_angle or timestamp"` |
| ジャイロバイアス | `gyro_bias.py` | `uv run pytest tests/test_pdr_regressions.py -k "gyro_bias"` |
| ステップ検出 | `step_detection.py` | `uv run pytest tests/test_pdr_regressions.py -k "step or detection or contact"` |
| 歩幅推定 | `step_length.py` | `uv run pytest tests/test_smoke.py tests/test_pdr_regressions.py -k "weinberg or step_length"` |
| 方位推定 | `heading.py` | `uv run pytest tests/test_pdr_regressions.py -k "heading or orientation or acceleration"` |
| 横歩き・平滑化 | `sidestep.py` | `uv run pytest tests/test_pdr_regressions.py tests/test_smoke.py -k "sidestep or movement_type or smooth"` |
| 軌跡・共有ステップ | `trajectory.py`, `models.py` | `uv run pytest tests/test_pdr_regressions.py -k "trajectory or prepare_pdr_steps or step_motion"` |
| CSV・pipeline | `outputs.py`, `pipeline.py` | `uv run pytest tests/test_pdr_regressions.py -k "dataframe or output or run"` |
| CLI・既定値 | `rikka/__init__.py`, `config.py` | `uv run pytest tests/test_smoke.py tests/test_pdr_regressions.py -k "help or default or validator"` |
| particle bridge | `particle_api.py` | `uv run pytest tests/test_pdr_regressions.py -k "prepare_pdr_steps or particle"` |

複数領域へまたがる変更では条件を無理に結合せず、関連する絞り込みを複数回実行する。
