"""新PFの全候補BLE因子、重みの一回適用、欠測互換性を検証する。"""

from itertools import product
from types import SimpleNamespace
from typing import cast

import matplotlib.image as mpimg
import numpy as np
import pytest

from rikka.common.lib.models import (
    Landmark,
    LandmarkDetection,
    LandmarkRange,
    PathLossModel,
    StepHeading,
)
from rikka.landmark.lib.assignment import build_step_observation_map
from rikka.particle.lib.candidate_bank import select_candidate_bank
from rikka.particle.lib.landmark import landmark_likelihood, landmark_range_likelihood
from rikka.particle.lib.runner import run_particle_steps
from rikka.particle.lib.weighting import weight_candidate_bank
from rikka.pdr.lib.motion_state.evidence import (
    build_particle_motion_headings,
    build_step_motion_evidences,
)


class _FixedDraw:
    def __init__(self, value):
        self.value = value

    def random(self, size):
        return np.full(size, self.value)


@pytest.fixture
def arguments(tmp_path):
    path = tmp_path / "map.png"
    mpimg.imsave(path, np.ones((81, 81)), cmap="gray", vmin=0, vmax=1)
    headings = [
        StepHeading(
            index,
            float(index - 1),
            0.0,
            None,
            None,
            0.0,
            "trajectory_motion",
            1.0,
            None,
            None,
            None,
            None,
            None,
            None,
            0.0,
            0.0,
            "forward",
            1.0,
            0.0,
            1.0,
            None,
            "forward",
        )
        for index in range(1, 5)
    ]
    return {
        "gx_mean": 0.0,
        "gz_mean": 9.8,
        "floormap_path": path,
        "origin_px": (40, 40),
        "scale": 1.0,
        "n_particles": 24,
        "prepared_step_headings": headings,
        "prepared_step_lengths": [1.0] * 4,
        "prepared_step_times": [float(index) for index in range(4)],
        "prepared_motion_evidences": build_step_motion_evidences(headings),
        "prepared_particle_motion_headings": build_particle_motion_headings(headings),
        "seed": 42,
        "landmark_mode": "observation",
        "landmarks": (Landmark("b", 43, 40),),
        "resample_ess_ratio": 0.001,
    }


def _assert_same_result(first, second):
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:3] == second[1:3]
    np.testing.assert_array_equal(first[3], second[3])
    assert first[4] == second[4]


def test_two_by_two_ble_weighting_identity_and_no_double_counting():
    # 候補分布F={0,1}/2を全列挙。G0と独立なBLE因子Lを一度掛けた
    # 非正規化事後質量 E_F[G0*L] が戻ることを選択区間の厳密和で確認する。
    base, likelihood = np.array([0.2, 0.8]), np.array([0.9, 0.1])
    expected = np.zeros(2)
    for candidates in product(range(2), repeat=2):
        factors = base[list(candidates)] * likelihood[list(candidates)]
        probabilities = factors / factors.sum()
        boundaries = np.concatenate(([0.0], np.cumsum(probabilities)))
        for index, probability in enumerate(probabilities):
            draw = float((boundaries[index] + boundaries[index + 1]) / 2)
            selected, means = select_candidate_bank(
                factors[None, :], cast(np.random.Generator, _FixedDraw(draw))
            )
            mass = weight_candidate_bank(np.ones(1), means)[0]
            expected[candidates[selected[0]]] += 0.25 * probability * mass
    np.testing.assert_allclose(expected, 0.5 * base * likelihood)
    assert not np.allclose(expected, 0.5 * base * likelihood**2)


@pytest.mark.parametrize("mode", ["observation", "ranging"])
def test_single_candidate_ble_equals_no_relocation(arguments, mode):
    kwargs = {
        **arguments,
        "landmark_mode": mode,
        "landmark_detections": (LandmarkDetection(0, "b", -59),),
        "landmark_ranging_observations": (LandmarkRange(0, "b", -59, 1, 1, -59),),
    }
    first, second = [], []
    a = run_particle_steps(
        **kwargs,
        proposal_method="candidate-bank",
        candidate_count=1,
        stage_collector=first,
    )
    b = run_particle_steps(
        **kwargs,
        proposal_method="no-relocation",
        stage_collector=second,
    )
    _assert_same_result(a, b)
    for bank_stage, single_stage in zip(first, second, strict=True):
        np.testing.assert_array_equal(
            bank_stage.posterior_weights, single_stage.posterior_weights
        )


@pytest.mark.parametrize("mode", ["observation", "ranging"])
def test_all_candidates_ble_factor_and_parent_weight_match_model(arguments, mode):
    stages = []
    model = PathLossModel(-59, 2, 6)
    kwargs = {
        **arguments,
        "landmark_mode": mode,
        "landmarks": (Landmark("b", 43, 40, path_loss_model=model),),
        "landmark_detections": (LandmarkDetection(0, "b", -59),),
        "landmark_ranging_observations": (LandmarkRange(0, "b", -59, 1, 1, -59),),
    }
    run_particle_steps(
        **kwargs,
        proposal_method="candidate-bank",
        candidate_count=8,
        stage_collector=stages,
    )
    first = stages[0]
    bank = first.candidate_bank
    assert bank is not None
    assert bank.base_factors is not None
    assert bank.landmark_factors is not None
    likelihood_function = (
        (lambda xy: landmark_likelihood(xy, (3, 0), 1, 0.05))
        if mode == "observation"
        else (lambda xy: landmark_range_likelihood(xy, (3, 0), -59, model, 0.05))
    )
    expected = np.stack(
        [likelihood_function(bank.positions[:, index]) for index in range(8)], axis=1
    )
    np.testing.assert_allclose(bank.landmark_factors, expected)
    np.testing.assert_allclose(bank.factors, bank.base_factors * expected)
    np.testing.assert_allclose(
        first.posterior_weights, first.before_weights * bank.factors.mean(axis=1)
    )
    # 欠測歩には同じ観測を持ち越さない。
    second = stages[1].candidate_bank
    assert second is not None
    np.testing.assert_array_equal(second.factors, second.base_factors)


@pytest.mark.parametrize("mode", ["observation", "ranging"])
def test_missing_or_unregistered_ble_does_not_change_bank_rng(arguments, mode):
    common = {
        **arguments,
        "landmark_mode": mode,
        "proposal_method": "candidate-bank",
        "candidate_count": 8,
    }
    baseline, unknown = [], []
    a = run_particle_steps(**common, stage_collector=baseline)
    b = run_particle_steps(
        **common,
        landmark_detections=(LandmarkDetection(0, "unknown", -50),),
        landmark_ranging_observations=(LandmarkRange(0, "unknown", -50, 1, 1, -50),),
        stage_collector=unknown,
    )
    _assert_same_result(a, b)
    for left, right in zip(baseline, unknown, strict=True):
        np.testing.assert_array_equal(
            left.candidate_bank.positions, right.candidate_bank.positions
        )
        np.testing.assert_array_equal(
            left.candidate_bank.factors, right.candidate_bank.factors
        )


def test_ranging_deduplicates_beacon_per_step_and_keeps_multiple_beacons():
    rows = (
        LandmarkRange(0.1, "a", -60, 1, 1, -60),
        LandmarkRange(0.2, "a", -80, 1, 1, -80),
        LandmarkRange(0.2, "b", -65, 1, 1, -65),
        LandmarkRange(0.2, "unknown", -40, 1, 1, -40),
        LandmarkRange(3, "a", -40, 1, 1, -40),
    )
    assigned = build_step_observation_map(rows, [1, 2], {"a": (0, 0), "b": (2, 0)})
    assert set(assigned) == {1}
    assert [item.beacon_id for item in assigned[1]] == ["a", "b"]
    assert [item.smoothed_rssi_dbm for item in assigned[1]] == [-70, -65]


def test_multiple_beacons_ranging_is_one_product_not_peak_times_product():
    from rikka.particle.lib.soft_landmark import soft_landmark_likelihood

    positions = np.array([[0, 1], [1, 1], [2, 1]], dtype=float)
    model = PathLossModel(-59, 2, 6)
    observations = (
        LandmarkRange(0, "a", -59, 1, 1, -59),
        LandmarkRange(0, "b", -65, 2, 1, -65),
    )
    ctx = SimpleNamespace(
        landmark_mode="ranging",
        step_number=1,
        landmark_by_step={1: LandmarkDetection(0, "a", -59)},
        landmark_observations_by_step={1: observations},
        landmark_definitions={
            key: Landmark(key, 0, 0, path_loss_model=model) for key in ("a", "b")
        },
        landmark_meters={"a": (0, 0), "b": (2, 0)},
        landmark_likelihood_floor=0.05,
        landmark_range_weight_power=0.5,
        landmark_sigma_m=1.0,
    )
    expected = np.sqrt(
        landmark_range_likelihood(positions, (0, 0), -59, model, 0.05)
        * landmark_range_likelihood(positions, (2, 0), -65, model, 0.05)
    )
    np.testing.assert_allclose(soft_landmark_likelihood(ctx, positions), expected)
    ctx.step_number = 2
    assert soft_landmark_likelihood(ctx, positions) is None


def test_recovery_weights_use_recovered_positions_once():
    from rikka.particle.lib.evaluate_map import _apply_recovery_landmark_observation

    positions = np.array([[0, 0], [1, 0], [3, 0]], dtype=float)
    prior = np.array([0.2, 0.3, 0.5])
    scores = np.array([-2.0, -1.0, -3.0])
    ctx = SimpleNamespace(
        proposal_method="candidate-bank",
        landmark_mode="observation",
        step_number=1,
        landmark_by_step={1: LandmarkDetection(0, "a", -50)},
        landmark_observations_by_step={},
        landmark_meters={"a": (0, 0)},
        landmark_sigma_m=1.0,
        landmark_likelihood_floor=0.05,
        particles=positions,
        weights=prior.copy(),
        next_path_log_scores=scores.copy(),
        # 破棄した通常候補の尤度を再利用してはいけない。
        landmark_likelihood=np.full(3, 0.001),
        landmark_update_count=1,
    )
    likelihood = landmark_likelihood(positions, (0, 0), 1, 0.05)
    expected = prior * likelihood
    expected /= expected.sum()
    _apply_recovery_landmark_observation(ctx)
    np.testing.assert_allclose(ctx.weights, expected)
    np.testing.assert_allclose(ctx.next_path_log_scores, scores + np.log(likelihood))
    assert ctx.landmark_update_count == 1
    assert ctx.landmark_observation_count == 1


def test_ranging_zero_power_is_neutral_even_with_peak_detection():
    from rikka.particle.lib.soft_landmark import soft_landmark_likelihood

    ctx = SimpleNamespace(
        landmark_mode="ranging",
        step_number=1,
        landmark_by_step={1: LandmarkDetection(0, "a", -50)},
        landmark_observations_by_step={1: (LandmarkRange(0, "a", -50, 1, 1, -50),)},
        landmark_range_weight_power=0.0,
    )
    assert soft_landmark_likelihood(ctx, np.array([[0, 0]], dtype=float)) is None


@pytest.mark.parametrize("mode", ["hybrid", "reset"])
def test_candidate_bank_rejects_unimplemented_soft_modes(arguments, mode):
    with pytest.raises(ValueError, match="ranging/observation"):
        run_particle_steps(
            **{**arguments, "landmark_mode": mode},
            proposal_method="candidate-bank",
            candidate_count=8,
            landmark_detections=(LandmarkDetection(0, "b", -50),),
        )


def test_candidate_bank_observation_rejects_legacy_gaussian_anchor(arguments):
    with pytest.raises(ValueError, match="hard_positions"):
        run_particle_steps(
            **{
                **arguments,
                "landmarks": (Landmark("b", 43, 40, position_sigma_m=0.5),),
            },
            proposal_method="candidate-bank",
            candidate_count=8,
            landmark_detections=(LandmarkDetection(0, "b", -50),),
        )


def _replay_context(mode, *, alpha=1, hard_step=0, checkpoints=(0,)):
    history = [np.zeros((3, 2))] * 3
    heading = SimpleNamespace(selected_heading=0.0)
    return SimpleNamespace(
        step_number=3,
        step_lengths=[1.0, 1.0],
        last_hard_anchor_step=hard_step,
        proposal_method="candidate-bank",
        landmark_mode=mode,
        landmark_range_weight_power=alpha,
        landmark_observations_by_step={1: (LandmarkRange(0, "a", -50, 1, 1, -50),)},
        landmark_by_step={1: LandmarkDetection(0, "a", -50)},
        healthy_checkpoint_steps=list(checkpoints),
        step_headings=[heading, heading],
        step_heading=heading,
        sl_det=1.0,
        position_history=history,
        heading_correction_history=history,
        heading_drift_history=history,
        stride_scale_history=history,
        weight_history=history,
        motion_state_history=history,
        n_particles=3,
        map_gray=np.ones((10, 10)),
        gx_mean=0.0,
        gz_mean=9.8,
        origin_px=(5, 5),
        scale=1.0,
        recovery_heading_sigma=0.2,
        rng=np.random.default_rng(0),
        adaptive_recovery_scale=False,
        effective_stride_scale_min=0.5,
        effective_stride_scale_max=1.5,
        recorder=SimpleNamespace(stages_enabled=False),
        recovery_attempts=0,
    )


@pytest.mark.parametrize("mode", ["observation", "ranging"])
def test_checkpoint_replay_cannot_erase_previous_ble(mode, monkeypatch):
    from rikka.particle.lib.recovery import apply

    ctx = _replay_context(mode)
    calls = []
    monkeypatch.setattr(
        apply, "_replay_from_checkpoint", lambda *a, **k: calls.append(a)
    )
    assert not apply._try_checkpoint_replay(ctx)
    assert ctx.checkpoint_step is None
    assert ctx.landmark_replay_blocked
    assert not calls


def test_zero_ranging_power_does_not_block_checkpoint_replay(monkeypatch):
    from rikka.particle.lib.recovery import apply

    ctx = _replay_context("ranging", alpha=0)
    calls = []
    monkeypatch.setattr(
        apply, "_replay_from_checkpoint", lambda *a, **k: calls.append(a)
    )
    apply._try_checkpoint_replay(ctx)
    assert ctx.checkpoint_step == 0
    assert not ctx.landmark_replay_blocked
    assert len(calls) == 1


def test_checkpoint_after_observation_can_be_replayed(monkeypatch):
    from rikka.particle.lib.recovery import apply

    ctx = _replay_context("ranging", checkpoints=(1, 0))
    calls = []
    monkeypatch.setattr(
        apply, "_replay_from_checkpoint", lambda *a, **k: calls.append(a)
    )
    apply._try_checkpoint_replay(ctx)
    assert ctx.checkpoint_step == 1
    assert len(calls) == 1


def test_checkpoint_replay_cannot_cross_hard_anchor_even_without_ble(monkeypatch):
    from rikka.particle.lib.recovery import apply

    ctx = _replay_context("none", hard_step=1)
    calls = []
    monkeypatch.setattr(
        apply, "_replay_from_checkpoint", lambda *a, **k: calls.append(a)
    )
    assert not apply._try_checkpoint_replay(ctx)
    assert ctx.checkpoint_step is None
    assert not calls
