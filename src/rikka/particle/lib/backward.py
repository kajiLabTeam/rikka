"""前向きで確定した歩列を後方PFの入力へ変換する。

役割:
    方位推定を再実行せず、歩順・絶対方位・観測時刻を反転する。
依存元:
    common.lib.models の共有型、dataclasses、NumPy を使用する。
利用先:
    particle.pipeline が独立した後方PFの準備と代表軌跡の並べ替えに使う。
処理フロー:
    歩時刻とBLE時刻に同じアフィン変換を適用し、絶対方位だけをπ回す。
    左右ラベル・尤度・重力平均は保持し、返却時に軌跡の歩順を戻す。
    stride_scale のAR(1)とcandidate-bankは逆順でも使えるが、運動状態の
    stay確率に基づく遷移行列は厳密な逆向きマルコフ連鎖ではない。
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import cast, overload

import numpy as np

from ...common.lib.models import LandmarkObservation, PreparedPdrSteps


def reverse_step_time(t: float, t_first: float, t_last: float) -> float:
    """歩と観測に共通の時刻反転を適用する。"""
    return float(t_first + t_last - t)


@overload
def _rotate(heading: float) -> float: ...


@overload
def _rotate(heading: None) -> None: ...


def _rotate(heading: float | None) -> float | None:
    """欠測を保持し、絶対方位を半回転して正規化する。"""
    if heading is None:
        return None
    angle = heading + np.pi
    wrapped = float(np.arctan2(np.sin(angle), np.cos(angle)))
    return -float(np.pi) if wrapped >= np.pi else wrapped


def build_backward_prepared(prepared: PreparedPdrSteps) -> PreparedPdrSteps:
    """入力を変更せず、確定済み歩情報の逆順コピーを返す。"""
    first = prepared.t_at_steps[0] if prepared.t_at_steps else 0.0
    last = prepared.t_at_steps[-1] if prepared.t_at_steps else 0.0
    times = [reverse_step_time(t, first, last) for t in reversed(prepared.t_at_steps)]
    headings = [
        heading._replace(
            step_index=i,
            timestamp_s=times[i - 1],
            gyro_heading=_rotate(heading.gyro_heading),
            accel_method1_heading=_rotate(heading.accel_method1_heading),
            accel_method2_heading=_rotate(heading.accel_method2_heading),
            selected_heading=_rotate(heading.selected_heading),
            body_heading=_rotate(heading.body_heading),
            motion_heading=_rotate(heading.motion_heading),
        )
        for i, heading in enumerate(reversed(prepared.step_headings), 1)
    ]
    return replace(
        prepared,
        trajectory=backward_to_forward(prepared.trajectory),
        step_lengths=list(reversed(prepared.step_lengths)),
        t_at_steps=times,
        step_headings=headings,
        particle_motion_headings=tuple(
            _rotate(h) for h in reversed(prepared.particle_motion_headings)
        ),
        motion_evidences=tuple(reversed(prepared.motion_evidences)),
        length_observations=tuple(
            obs._replace(step_index=i)
            for i, obs in enumerate(reversed(prepared.length_observations), 1)
        ),
        motion_observations=tuple(
            obs._replace(
                step_index=i,
                timestamp_s=reverse_step_time(obs.timestamp_s, first, last),
                device_yaw_heading=_rotate(obs.device_yaw_heading),
                body_heading_candidate=_rotate(obs.body_heading_candidate),
                directed_motion_heading=_rotate(obs.directed_motion_heading),
                motion_axis_heading=_rotate(obs.motion_axis_heading),
            )
            for i, obs in enumerate(reversed(prepared.motion_observations), 1)
        ),
        motion_posteriors=tuple(
            obs._replace(step_index=i, heading_mean=float(_rotate(obs.heading_mean)))
            for i, obs in enumerate(reversed(prepared.motion_posteriors), 1)
        ),
        direction_posteriors=tuple(
            replace(
                obs, step_index=i, selected_heading=float(_rotate(obs.selected_heading))
            )
            for i, obs in enumerate(reversed(prepared.direction_posteriors), 1)
        ),
    )


def reverse_observation_times[Observation: LandmarkObservation](
    observations: Sequence[Observation], t_first: float, t_last: float
) -> tuple[Observation, ...]:
    """観測の時刻だけを反転し、増加順に並べた新しいタプルを返す。"""
    return tuple(
        sorted(
            (
                cast(
                    Observation,
                    obs._replace(
                        timestamp_s=reverse_step_time(obs.timestamp_s, t_first, t_last)
                    ),
                )
                for obs in observations
            ),
            key=lambda obs: obs.timestamp_s,
        )
    )


def backward_to_forward(trajectory: Sequence[Sequence[float]]) -> list[list[float]]:
    """座標系を変えず、原点ノードを含む後方経路の順序だけを戻す。"""
    return [list(point) for point in reversed(trajectory)]
