"""周期が変化する合成歩行信号で適応歩検出のイベントと診断を検証する。"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from rikka import cli
from rikka.cli import commands, options
from rikka.common.lib import sensors
from rikka.pdr.lib.step_detection import detect_step_result


def _signal(
    events: np.ndarray,
    *,
    sample_rate: float = 100.0,
    irregular: bool = False,
    secondary: bool = False,
) -> pd.DataFrame:
    """既知の歩時刻から、主ピークと任意の弱い副ピークを生成する。"""
    duration = float(events[-1] + 2.0) if len(events) else 8.0
    times = np.arange(0.0, duration, 1.0 / sample_rate)
    if irregular:
        rng = np.random.default_rng(14)
        increments = (1.0 / sample_rate) * rng.uniform(0.7, 1.3, len(times))
        times = np.cumsum(increments)
        times -= times[0]
    values = 0.12 + 0.02 * np.sin(2.0 * np.pi * 7.3 * times)
    for index, event in enumerate(events):
        amplitude = 3.2 + 0.4 * (index % 2)
        values += amplitude * np.exp(-0.5 * ((times - event) / 0.035) ** 2)
        if secondary:
            values += 0.5 * np.exp(-0.5 * ((times - event - 0.17) / 0.03) ** 2)
    return pd.DataFrame({"t": times, "low_lin_norm": values})


def _assert_matches_events(
    frame: pd.DataFrame, events: np.ndarray, tolerance_s: float = 0.04
):
    """総数だけでなく、採用した各ピークが実イベントに対応することを確認する。"""
    result = detect_step_result(frame, method="adaptive_peak")
    detected = frame["t"].to_numpy()[result.peaks]
    assert result.method == "adaptive_peak"
    assert len(detected) == len(events)
    np.testing.assert_allclose(detected, events, atol=tolerance_s, rtol=0.0)
    return result


def test_jitter_near_legacy_interval_preserves_real_steps():
    intervals = np.tile([0.46, 0.59, 0.48, 0.60], 8)
    events = np.r_[2.0, 2.0 + np.cumsum(intervals)]
    frame = _signal(events)

    legacy = detect_step_result(frame, method="peak")
    assert len(legacy.peaks) < len(events)
    _assert_matches_events(frame, events)


@pytest.mark.parametrize("cadence_hz", [1.2, 1.8, 2.5])
def test_regular_walking_across_supported_cadences(cadence_hz):
    events = np.arange(2.0, 17.0, 1.0 / cadence_hz)
    _assert_matches_events(_signal(events), events)


def test_cadence_change_and_pause_do_not_create_or_drop_events():
    slow = np.arange(2.0, 10.0, 1.0 / 1.2)
    fast = np.arange(13.0, 23.0, 1.0 / 2.5)
    events = np.r_[slow, fast]
    _assert_matches_events(_signal(events), events)


def test_continuous_acceleration_and_deceleration_preserve_event_timing():
    intervals = np.r_[np.linspace(0.80, 0.40, 18), np.linspace(0.40, 0.80, 18)]
    events = np.r_[2.0, 2.0 + np.cumsum(intervals)]
    _assert_matches_events(_signal(events), events)


def test_weak_secondary_peaks_are_not_counted_as_additional_steps():
    events = np.arange(2.0, 17.0, 0.53)
    _assert_matches_events(_signal(events, secondary=True), events)


@pytest.mark.parametrize("secondary_height", [1.2, 1.5])
def test_secondary_peaks_above_height_threshold_are_suppressed(secondary_height):
    events = np.arange(2.0, 17.0, 0.53)
    frame = _signal(events)
    times = frame["t"].to_numpy()
    for event in events:
        frame["low_lin_norm"] += secondary_height * np.exp(
            -0.5 * ((times - event - 0.17) / 0.03) ** 2
        )
    result = _assert_matches_events(frame, events)
    assert len(result.diagnostics) > len(result.peaks)
    assert any(not item.accepted for item in result.diagnostics)


@pytest.mark.parametrize("sample_count", [0, 1, 2])
def test_empty_and_too_short_signals_have_no_steps(sample_count):
    frame = pd.DataFrame(
        {
            "t": np.arange(sample_count) * 0.01,
            "low_lin_norm": np.full(sample_count, 0.12),
        }
    )
    result = detect_step_result(frame, method="adaptive_peak")
    assert result.method == "adaptive_peak"
    assert len(result.peaks) == 0
    assert result.diagnostics == ()


def test_long_timestamp_gap_does_not_become_an_interpolated_period():
    events = np.arange(2.0, 13.0, 0.5)
    frame = _signal(events)
    frame = frame.loc[~frame["t"].between(5.0, 6.0, inclusive="left")].reset_index(
        drop=True
    )
    recorded_events = events[(events < 5.0) | (events >= 6.0)]
    result = _assert_matches_events(frame, recorded_events)
    near_gap = [item for item in result.diagnostics if 4.0 <= item.timestamp_s <= 7.0]
    assert near_gap
    assert all(item.period_s is None for item in near_gap)


@pytest.mark.parametrize("baseline", [0.12, 1.2])
def test_stationary_low_activity_does_not_create_steps(baseline):
    times = np.arange(0.0, 12.0, 0.01)
    values = baseline + 0.03 * np.sin(2.0 * np.pi * 1.9 * times)
    frame = pd.DataFrame({"t": times, "low_lin_norm": values})
    result = detect_step_result(frame, method="adaptive_peak")
    assert len(result.peaks) == 0
    assert not any(item.accepted for item in result.diagnostics)


@pytest.mark.parametrize("sample_rate", [50.0, 100.0, 160.0])
@pytest.mark.parametrize("irregular", [False, True])
def test_detection_uses_elapsed_time_across_sampling_patterns(sample_rate, irregular):
    events = np.arange(2.0, 16.0, 0.43)
    frame = _signal(events, sample_rate=sample_rate, irregular=irregular)
    _assert_matches_events(frame, events)


@pytest.mark.parametrize("invalid_time", ["duplicate", "reverse", "nan", "inf"])
def test_invalid_timestamps_are_rejected(invalid_time):
    frame = _signal(np.arange(2.0, 8.0, 0.5))
    if invalid_time == "duplicate":
        frame.loc[20, "t"] = frame.loc[19, "t"]
    elif invalid_time == "reverse":
        frame.loc[20, "t"] = frame.loc[19, "t"] - 0.01
    else:
        frame.loc[20, "t"] = float(invalid_time)
    with pytest.raises(ValueError):
        detect_step_result(frame, method="adaptive_peak")


def test_nonfinite_signal_is_rejected():
    frame = _signal(np.arange(2.0, 8.0, 0.5))
    frame.loc[20, "low_lin_norm"] = np.nan
    with pytest.raises(ValueError):
        detect_step_result(frame, method="adaptive_peak")


def test_diagnostics_account_for_selected_and_rejected_candidates():
    events = np.arange(2.0, 15.0, 0.43)
    frame = _signal(events)
    # 高さ条件は満たすが突出が小さい振動を、歩行前の静止区間に置く。
    stationary = frame["t"] < 1.2
    frame.loc[stationary, "low_lin_norm"] = 1.2 + 0.03 * np.sin(
        2.0 * np.pi * 2.0 * frame.loc[stationary, "t"]
    )
    result = _assert_matches_events(frame, events)

    assert isinstance(result.diagnostics, tuple)
    accepted = [item.sample_index for item in result.diagnostics if item.accepted]
    assert accepted == list(result.peaks)
    assert any(not item.accepted for item in result.diagnostics)
    for item in result.diagnostics:
        assert item.timestamp_s == pytest.approx(frame["t"].iloc[item.sample_index])
        assert item.period_s is None or item.period_s > 0.0
        assert np.isfinite(item.confidence)
        assert 0.0 <= item.confidence <= 1.0
        assert item.min_interval_s > 0.0
        assert item.prominence >= 0.0
        assert item.reason


@pytest.mark.parametrize("command", ["run", "particle"])
def test_cli_forwards_adaptive_step_detection_to_shared_run(monkeypatch, command):
    monkeypatch.setattr(
        options, "_resolve_measurement_settings", Mock(return_value=((0, 0), 0.0, 1.7))
    )
    monkeypatch.setattr(
        options, "_resolve_ble_inputs", Mock(return_value=("unused.csv", ()))
    )
    acc, gyro = object(), object()
    monkeypatch.setattr(sensors, "load_sensor_data", Mock(return_value=(acc, gyro)))
    run = Mock()
    monkeypatch.setattr(commands, "run", run)

    result = CliRunner().invoke(
        cli, [command, "--step-detection", "adaptive_peak", "--no-plot"]
    )

    assert result.exit_code == 0, result.output
    run.assert_called_once()
    assert run.call_args.kwargs["step_detection_method"] == "adaptive_peak"
    assert run.call_args.kwargs["df_acc"] is acc
    assert run.call_args.kwargs["df_gyro"] is gyro
    assert run.call_args.kwargs["use_particle_filter"] is (command == "particle")
