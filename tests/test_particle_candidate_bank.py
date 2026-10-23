"""有限候補PFの重み補正、状態整合性、既存方式との互換性を検証する。"""

from dataclasses import fields
from importlib import import_module
from itertools import product
from typing import cast

import matplotlib.image as mpimg
import numpy as np
import pytest

from rikka.common.lib.models import LandmarkDetection, StepHeading
from rikka.particle.lib.candidate_bank import select_candidate_bank
from rikka.particle.lib.recorder import ParticleStepStages
from rikka.particle.lib.runner import run_particle_steps
from rikka.pdr.lib.motion_state.evidence import (
    build_particle_motion_headings,
    build_step_motion_evidences,
)


class _FixedDraw:
    """離散選択の確率区間を厳密に列挙する固定乱数。"""

    def __init__(self, value: float) -> None:
        self.value = value

    def random(self, size):
        return np.full(size, self.value)


def _heading(index: int) -> StepHeading:
    return StepHeading(
        step_index=index,
        timestamp_s=float(index - 1),
        gyro_heading=0.0,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=0.0,
        source="trajectory_motion",
        confidence=1.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=0.0,
        motion_heading=0.0,
        movement_type="forward",
        forward_displacement=1.0,
        lateral_displacement=0.0,
        motion_confidence=1.0,
        motion_reject_reason=None,
        trajectory_movement_type="forward",
    )


@pytest.fixture
def arguments(tmp_path):
    path = tmp_path / "map.png"
    mpimg.imsave(path, np.ones((81, 81)), cmap="gray", vmin=0, vmax=1)
    headings = [_heading(index) for index in range(1, 5)]
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
    }


def _assert_same_result(first, second) -> None:
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:3] == second[1:3]
    np.testing.assert_array_equal(first[3], second[3])
    assert first[4] == second[4]


def _assert_same_stages(first, second, *, ignore_bank=False) -> None:
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        for field in fields(ParticleStepStages):
            if ignore_bank and field.name in {"candidate_bank", "proposal_method"}:
                continue
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            elif field.name != "candidate_bank":
                assert a == b
            elif a is not None:
                for bank_field in fields(a):
                    np.testing.assert_array_equal(
                        getattr(a, bank_field.name), getattr(b, bank_field.name)
                    )
            else:
                assert b is None


@pytest.mark.parametrize(
    "factors",
    [
        np.array([1.0, 3.0]),
        np.array([0.0, 2.0]),
        np.array([0.2, 0.8]),
    ],
)
def test_candidate_bank_proper_weighting_by_full_enumeration(factors) -> None:
    # F(0)=F(1)=1/2の全候補組と全選択区間を列挙する。
    # E[(S/K) h(X_J)] = E_F[G(X) h(X)] を確認し、Gの二重計上を検出する。
    expected_mass = np.zeros(2)
    for candidates in product(range(2), repeat=2):
        scores = factors[list(candidates)]
        total = float(scores.sum())
        if total == 0:
            selected, means = select_candidate_bank(
                scores[None, :], np.random.default_rng(0)
            )
            assert selected[0] == -1
            assert means[0] == 0
            continue
        boundaries = np.concatenate(([0.0], np.cumsum(scores / total)))
        for index in range(2):
            probability = boundaries[index + 1] - boundaries[index]
            if probability == 0:
                continue
            draw = (boundaries[index + 1] + boundaries[index]) / 2
            selected, means = select_candidate_bank(
                scores[None, :], cast(np.random.Generator, _FixedDraw(draw))
            )
            assert selected[0] == index
            expected_mass[candidates[index]] += 0.25 * probability * means[0]
    np.testing.assert_allclose(expected_mass, factors / 2)


def test_candidate_bank_retains_parent_acceptance_mass() -> None:
    selected, means = select_candidate_bank(
        np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]]),
        np.random.default_rng(3),
    )
    np.testing.assert_array_equal(means, [1, 0.25, 0])
    assert selected[1] == 0
    assert selected[2] == -1


def test_candidate_bank_single_candidate_consumes_no_rng() -> None:
    first, control = np.random.default_rng(8), np.random.default_rng(8)
    selected, means = select_candidate_bank(np.array([[0.2], [0], [1]]), first)
    np.testing.assert_array_equal(selected, [0, -1, 0])
    np.testing.assert_array_equal(means, [0.2, 0, 1])
    np.testing.assert_array_equal(first.random(20), control.random(20))


@pytest.mark.parametrize(
    "factors",
    [
        np.array([1, 2]),
        np.empty((2, 0)),
        np.ones((2, 1, 1)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[-0.1]]),
    ],
)
def test_candidate_bank_rejects_invalid_factors(factors) -> None:
    with pytest.raises(ValueError):
        select_candidate_bank(factors, np.random.default_rng(0))


def test_candidate_bank_avoids_overflow_of_finite_factors() -> None:
    selected, means = select_candidate_bank(
        np.full((2, 3), np.finfo(float).max), np.random.default_rng(0)
    )
    assert np.all(selected >= 0)
    np.testing.assert_array_equal(means, np.full(2, np.finfo(float).max))


def test_default_equals_explicit_legacy(arguments) -> None:
    first, second = [], []
    a = run_particle_steps(**arguments, stage_collector=first)
    b = run_particle_steps(
        **arguments, proposal_method="legacy", stage_collector=second
    )
    _assert_same_result(a, b)
    _assert_same_stages(first, second)


@pytest.mark.parametrize("wall", [False, True])
def test_single_candidate_equals_no_relocation(arguments, wall) -> None:
    if wall:
        image = np.ones((81, 81))
        image[:, 42] = 0
        mpimg.imsave(arguments["floormap_path"], image, cmap="gray", vmin=0, vmax=1)
    first, second = [], []
    a = run_particle_steps(
        **arguments, proposal_method="candidate-bank", stage_collector=first
    )
    b = run_particle_steps(
        **arguments, proposal_method="no-relocation", stage_collector=second
    )
    _assert_same_result(a, b)
    _assert_same_stages(first, second, ignore_bank=True)


def test_candidate_bank_is_reproducible_and_diagnostics_do_not_change_rng(arguments):
    common = {**arguments, "proposal_method": "candidate-bank", "candidate_count": 5}
    first, second, diagnostics = [], [], []
    a = run_particle_steps(**common, stage_collector=first)
    b = run_particle_steps(
        **common, stage_collector=second, diagnostics_collector=diagnostics
    )
    c = run_particle_steps(**common)
    _assert_same_result(a, b)
    _assert_same_result(a, c)
    _assert_same_stages(first, second)
    assert len(diagnostics) == 4


def test_single_candidate_zero_factor_preserves_map_validity(arguments) -> None:
    # 許容される大きなpowerで因子をunderflowさせる。地図として有効な辺を
    # 因子0だけを理由に無効化すると、K=1の診断互換性が失われる。
    common = {**arguments, "motion_predictive_weight_power": 1000000}
    first, second, first_diagnostics, second_diagnostics = [], [], [], []
    a = run_particle_steps(
        **common,
        proposal_method="candidate-bank",
        stage_collector=first,
        diagnostics_collector=first_diagnostics,
    )
    b = run_particle_steps(
        **common,
        proposal_method="no-relocation",
        stage_collector=second,
        diagnostics_collector=second_diagnostics,
    )
    bank = first[0].candidate_bank
    assert bank is not None
    assert np.all(bank.factors == 0)
    assert np.all(bank.valid_transition)
    assert np.all(bank.selected_indices == -1)
    _assert_same_result(a, b)
    _assert_same_stages(first, second, ignore_bank=True)
    assert first_diagnostics == second_diagnostics


def test_candidate_bank_selected_state_and_weight_are_consistent(arguments) -> None:
    stages = []
    run_particle_steps(
        **arguments,
        proposal_method="candidate-bank",
        candidate_count=5,
        stage_collector=stages,
        resample_ess_ratio=0.001,
    )
    for stage in stages:
        bank = stage.candidate_bank
        assert bank is not None
        rows = np.arange(len(bank.selected_indices))
        selected = bank.selected_indices
        assert np.all(selected >= 0)
        np.testing.assert_array_equal(
            stage.proposed_positions, bank.positions[rows, selected]
        )
        np.testing.assert_array_equal(
            stage.proposed_headings, bank.headings[rows, selected]
        )
        np.testing.assert_array_equal(
            stage.proposed_step_lengths, bank.step_lengths[rows, selected]
        )
        np.testing.assert_array_equal(
            stage.proposed_motion_state, bank.motion_state[rows, selected]
        )
        np.testing.assert_allclose(
            stage.posterior_weights, stage.before_weights * bank.mean_factors
        )
        displacement = stage.proposed_positions - stage.before_positions
        np.testing.assert_allclose(
            np.linalg.norm(displacement, axis=1), stage.proposed_step_lengths
        )
        np.testing.assert_allclose(
            displacement[:, 0],
            stage.proposed_step_lengths * np.cos(stage.proposed_headings),
        )
        np.testing.assert_allclose(
            displacement[:, 1],
            stage.proposed_step_lengths * np.sin(stage.proposed_headings),
        )
        assert stage.recovery_mode == "none"
        assert not stage.resampled
        np.testing.assert_array_equal(
            stage.after_stride_scale, bank.stride_scale[rows, selected]
        )
        # この開放地図では絶対方位補正・rejuvenationは起こらず、採用した
        # driftが次歩へそのまま引き継がれる。位置だけの救済を検出する。
        np.testing.assert_allclose(
            stage.after_offsets, bank.heading_drift[rows, selected]
        )


@pytest.mark.parametrize(
    "method,count",
    [
        ("unknown", 1),
        ("candidate-bank", 0),
        ("candidate-bank", -1),
        ("candidate-bank", 1.5),
        ("candidate-bank", True),
        ("legacy", 2),
        ("no-relocation", 2),
    ],
)
def test_invalid_proposal_settings_fail(arguments, method, count) -> None:
    with pytest.raises(ValueError):
        run_particle_steps(**arguments, proposal_method=method, candidate_count=count)


def test_candidate_bank_rejects_actual_ble_observation(arguments) -> None:
    with pytest.raises(ValueError, match="BLE|ランドマーク|landmark"):
        run_particle_steps(
            **arguments,
            proposal_method="candidate-bank",
            candidate_count=3,
            landmark_detections=(LandmarkDetection(0, "beacon", -50),),
        )


def test_candidate_bank_invalid_transitions_have_zero_selection_mass(arguments) -> None:
    image = np.ones((81, 81))
    image[:, 42] = 0
    mpimg.imsave(arguments["floormap_path"], image, cmap="gray", vmin=0, vmax=1)
    stages = []
    run_particle_steps(
        **arguments,
        proposal_method="candidate-bank",
        candidate_count=5,
        stage_collector=stages,
        sigma_init_heading=0.5,
    )
    assert any(np.any(~stage.candidate_bank.valid_transition) for stage in stages)
    for stage in stages:
        bank = stage.candidate_bank
        assert bank is not None
        assert np.all(bank.factors[~bank.valid_transition] == 0)
        selected = bank.selected_indices
        live = selected >= 0
        assert np.all(bank.valid_transition[np.flatnonzero(live), selected[live]])
        assert np.all(bank.mean_factors[~live] == 0)


def test_all_invalid_candidates_use_existing_recovery(arguments) -> None:
    image = np.zeros((81, 81))
    image[40, 40] = 1
    mpimg.imsave(arguments["floormap_path"], image, cmap="gray", vmin=0, vmax=1)
    stages, diagnostics = [], []
    run_particle_steps(
        **arguments,
        proposal_method="candidate-bank",
        candidate_count=3,
        stage_collector=stages,
        diagnostics_collector=diagnostics,
        sigma_sl_ratio=0,
        stride_scale_init_sigma=0,
        stride_scale_process_sigma=0,
    )
    bank = stages[0].candidate_bank
    assert bank is not None
    assert not np.any(bank.valid_transition)
    np.testing.assert_array_equal(bank.selected_indices, -np.ones(24, dtype=int))
    np.testing.assert_array_equal(bank.mean_factors, np.zeros(24))
    assert diagnostics[0].recovery_attempted
    assert diagnostics[0].recovery_mode != "none"
    assert np.isfinite(stages[0].after_weights).all()
    assert stages[0].after_weights.sum() == pytest.approx(1)


def test_checkpoint_replay_preserves_original_bank_and_marks_history_version(
    arguments, monkeypatch
) -> None:
    runner = import_module("rikka.particle.lib.runner")
    evaluate_map = import_module("rikka.particle.lib.evaluate_map")
    original_propose = runner.propose
    original_banks = []

    def force_second_step_recovery(ctx):
        original_propose(ctx)
        original_banks.append(ctx.candidate_bank)
        if ctx.step_number == 2:
            # 開放地図でcheckpoint経路だけを独立して検証するための介入。
            # 候補自体は変えず、通常の集団復旧判定だけを強制する。
            ctx.valid_weight_mass = 0.0
            ctx.recovery_attempted = True

    monkeypatch.setattr(runner, "propose", force_second_step_recovery)
    monkeypatch.setattr(evaluate_map, "_try_local_recovery", lambda ctx: False)
    common = {**arguments, "proposal_method": "candidate-bank", "candidate_count": 5}
    first, second, diagnostics = [], [], []
    a = run_particle_steps(**common, stage_collector=first)
    b = run_particle_steps(
        **common, stage_collector=second, diagnostics_collector=diagnostics
    )
    _assert_same_result(a, b)
    _assert_same_stages(first, second)
    assert first[0].recovery_mode == "checkpoint_replayed"
    assert diagnostics[1].recovery_checkpoint_step == 0
    assert diagnostics[1].recovery_replay_steps == 2
    for index in (0, 1):
        stage = first[index]
        assert stage.candidate_bank is original_banks[index]
        assert stage.candidate_bank is not None
        assert stage.proposal_method == "candidate-bank"
        assert stage.proposal_history_version == 0
        assert stage.history_version == 1
        assert not stage.proposal_matches_final_history
    for index in (2, 3):
        stage = first[index]
        assert stage.candidate_bank is original_banks[index]
        assert stage.proposal_history_version == 1
        assert stage.history_version == 1
        assert stage.proposal_matches_final_history
