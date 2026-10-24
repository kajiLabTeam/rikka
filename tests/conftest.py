"""旧PDR/PFの数値回帰条件を、変更される製品既定値から分離する。

テスト全体のmatplotlibバックエンドを非対話の Agg に固定する役割も持つ。
描画を伴うテストが実行環境の既定バックエンドで表示ウィンドウを開くと、
`plt.show()` で停止したり図が残って失敗するため、importより前に設定する。
"""

import matplotlib

matplotlib.use("Agg")

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from rikka.particle import pipeline  # noqa: E402
from rikka.pdr.lib import preparation  # noqa: E402


@pytest.fixture(scope="module")
def legacy_particle_profile():
    """golden作成時（5807a74）のPF条件を明示し、期待値自体は変更しない。"""
    original = pipeline.run_particle_steps
    profile = {
        "sigma_init_heading": 0.03,
        "sigma_heading": 0.01,
        "sigma_sl_ratio": 0.01,
        "stride_scale_prior_mean": 1.03,
        "stride_scale_init_sigma": 0.05,
        "stride_scale_retention": 0.995,
        "stride_scale_process_sigma": 0.002,
        "stride_scale_rejuvenation_sigma": 0.003,
        "stride_scale_min": 0.90,
        "stride_scale_max": 1.15,
        "heading_drift_retention": 0.85,
        "rejuvenation_sigma_heading": 0.009,
        "invalid_relocation_radius_m": 0.0,
    }

    def fixed_steps(*args, **kwargs):
        return original(*args, **(kwargs | profile))

    # 旧参照値の係数も固定し、校正済みの製品既定値とは別に互換性を検証する。
    with (
        patch.object(pipeline, "run_particle_steps", fixed_steps),
        patch.object(
            preparation, "compute_weinberg_k", lambda height: 0.47 * (height / 1.70)
        ),
    ):
        yield
