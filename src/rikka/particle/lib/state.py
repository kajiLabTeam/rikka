"""particle filter の現在状態と履歴を同期管理する。

役割:
    並行して更新される6本の粒子配列と履歴系列を型付きオブジェクトへ集約する。
依存元:
    NumPy 配列と標準 dataclass だけを使用する。
利用先:
    particle pipeline の各段階と checkpoint replay から使用する。
処理フロー:
    不変な現在状態を受け取り、履歴へcopyを追加し、必要時に指定歩まで切り詰める。
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParticleState:
    """1歩時点の全粒子状態。"""

    positions: np.ndarray
    heading_correction: np.ndarray
    heading_drift: np.ndarray
    motion_state: np.ndarray
    stride_scale: np.ndarray
    weights: np.ndarray


@dataclass
class ParticleHistory:
    """粒子状態・スコア・親インデックスの同期履歴。"""

    positions: list[np.ndarray] = field(default_factory=list)
    heading_corrections: list[np.ndarray] = field(default_factory=list)
    heading_drifts: list[np.ndarray] = field(default_factory=list)
    motion_states: list[np.ndarray] = field(default_factory=list)
    stride_scales: list[np.ndarray] = field(default_factory=list)
    weights: list[np.ndarray] = field(default_factory=list)
    path_log_scores: list[np.ndarray] = field(default_factory=list)
    parents: list[np.ndarray] = field(default_factory=list)

    def append(
        self,
        state: ParticleState,
        path_log_scores: np.ndarray,
        parents: np.ndarray | None = None,
    ) -> None:
        """現在状態をcopyして全履歴へ同時に追加する。"""
        self.positions.append(state.positions.copy())
        self.heading_corrections.append(state.heading_correction.copy())
        self.heading_drifts.append(state.heading_drift.copy())
        self.motion_states.append(state.motion_state.copy())
        self.stride_scales.append(state.stride_scale.copy())
        self.weights.append(state.weights.copy())
        self.path_log_scores.append(path_log_scores.copy())
        if parents is not None:
            self.parents.append(parents.copy())

    def truncate_to(self, step: int) -> None:
        """指定歩を末尾として全状態履歴を同時に切り詰める。"""
        keep = step + 1
        self.positions[keep:] = []
        self.heading_corrections[keep:] = []
        self.heading_drifts[keep:] = []
        self.motion_states[keep:] = []
        self.stride_scales[keep:] = []
        self.weights[keep:] = []
        self.path_log_scores[keep:] = []
        self.parents[step:] = []

    def checkpoint(self, step: int) -> ParticleState:
        """指定歩の状態を復元可能なcopyとして返す。"""
        return ParticleState(
            self.positions[step].copy(),
            self.heading_corrections[step].copy(),
            self.heading_drifts[step].copy(),
            self.motion_states[step].copy(),
            self.stride_scales[step].copy(),
            self.weights[step].copy(),
        )

    def extend_replay(self, states: list[ParticleState]) -> None:
        """再生済み状態列を既存履歴の末尾へ追加する。"""
        for state in states:
            self.append(
                state,
                np.zeros(len(state.positions), dtype=float),
            )


@dataclass
class ParticleRuntime:
    """段階関数間で共有する1回のPF実行状態。"""

    values: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        """辞書に保存した実行値を属性として返す。"""
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        """定義済みフィールド以外を実行値辞書へ保存する。"""
        if name == "values":
            object.__setattr__(self, name, value)
        else:
            self.values[name] = value

    def update(self, values: dict[str, Any]) -> None:
        """複数の入力値を一括登録する。"""
        self.values.update(values)
