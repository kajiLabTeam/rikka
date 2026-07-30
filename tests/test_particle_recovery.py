"""particle 状態履歴と recovery 戦略の単体回帰。"""

import numpy as np

from rikka.particle.lib.recovery.fallback import HoldPosition
from rikka.particle.lib.recovery.protocol import RecoveryChain
from rikka.particle.lib.state import ParticleHistory, ParticleState


def _state(offset: float) -> ParticleState:
    positions = np.asarray([[offset, 0.0], [offset, 1.0]])
    values = np.asarray([offset, offset + 1.0])
    return ParticleState(
        positions,
        values,
        values,
        np.asarray([0, 1], dtype=np.int8),
        np.ones(2),
        np.full(2, 0.5),
    )


def test_particle_history_truncates_all_series_together() -> None:
    history = ParticleHistory()
    history.append(_state(0.0), np.zeros(2))
    history.append(_state(1.0), np.ones(2), np.asarray([0, 1]))
    history.append(_state(2.0), np.full(2, 2.0), np.asarray([1, 0]))

    history.truncate_to(1)

    assert len(history.positions) == 2
    assert len(history.heading_corrections) == 2
    assert len(history.path_log_scores) == 2
    assert len(history.parents) == 1
    np.testing.assert_array_equal(
        history.checkpoint(1).positions, _state(1.0).positions
    )


def test_recovery_chain_falls_back_to_hold_position() -> None:
    state = _state(0.0)
    outcome = RecoveryChain((HoldPosition(),)).recover(state)

    assert outcome is not None
    assert outcome.mode == "failed_hold"
    assert outcome.state is state
