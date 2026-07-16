import pytest
from click.testing import CliRunner

from rikka import cli
from rikka.config import (
    FORWARD_HEADING_SOURCE,
    HEADING_METHOD,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    WEINBERG_REFERENCE_K,
    compute_weinberg_k,
)
from rikka.ping import ping


def test_ping_returns_expected_message() -> None:
    assert ping() == "Hello, rikka"


def test_compute_weinberg_k_scales_with_height() -> None:
    assert compute_weinberg_k(1.70) == pytest.approx(WEINBERG_REFERENCE_K)
    assert compute_weinberg_k(1.80) > compute_weinberg_k(1.60)


def test_compute_weinberg_k_rejects_non_positive_height() -> None:
    with pytest.raises(ValueError, match="height_m"):
        compute_weinberg_k(0.0)


@pytest.mark.parametrize("height_m", [float("nan"), float("inf"), float("-inf")])
def test_compute_weinberg_k_rejects_non_finite_height(height_m: float) -> None:
    with pytest.raises(ValueError, match="有限"):
        compute_weinberg_k(height_m)


def test_default_sidestep_detection_settings_match_selected_standard() -> None:
    assert HEADING_METHOD == "gyro_accel_motion"
    assert FORWARD_HEADING_SOURCE == "body"
    assert SIDESTEP_LATERAL_RATIO == pytest.approx(1.2)
    assert SIDESTEP_MIN_LATERAL_DISPLACEMENT_M == pytest.approx(0.03)
    assert SIDESTEP_SMOOTHING_METHOD == "clustered"


def test_run_help_includes_forward_heading_source_option() -> None:
    result = CliRunner().invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--forward-heading-source" in result.output


def test_run_help_includes_adaptive_motion_options() -> None:
    result = CliRunner().invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--motion-estimation" in result.output
    assert "--smoothing" in result.output
    assert "legacy" in result.output


def test_particle_help_includes_pf_seed_option() -> None:
    result = CliRunner().invoke(cli, ["particle", "--help"])

    assert result.exit_code == 0
    assert "--pf-seed" in result.output


@pytest.mark.parametrize("command", ["run", "pdr", "particle", "sensor"])
def test_cli_manual_gyro_bias_requires_bias_value(command: str) -> None:
    result = CliRunner().invoke(cli, [command, "--gyro-bias-method", "manual"])

    assert result.exit_code != 0
    assert "--gyro-bias" in result.output
