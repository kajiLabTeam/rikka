"""旧PFの数値回帰に必要な条件を、変更される製品既定値から分離する。"""

from unittest.mock import patch

import pytest

from rikka.particle import pipeline


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

    with patch.object(pipeline, "run_particle_steps", fixed_steps):
        yield
