"""エージェント検証用: Ryuki 系データの方位ガード比較プロットを生成する。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rikka as rikka  # isort: skip

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rikka.analyze import pdr
from rikka.config import (
    FORWARD_HEADING_SOURCE,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    USER_HEIGHT_M,
    compute_weinberg_k,
)

DATASETS = (
    Path("input/ryuki_1turn_rightsidestep_3turn_leftsidestep3"),
    Path("input/ryuki_1turn_rightsidestep_3turn_leftsidestep4"),
)
OUTPUT_PATH = Path("output/diagnostics/ryuki_heading_fix_comparison.png")
MOTION_BODY_REJECT_DEG = 90.0
SUSPECT_HEADING_LIMIT_RAD = np.deg2rad(25.0)


@dataclass(frozen=True)
class Variant:
    name: str
    sidestep_suspect_mode: str = "motion"
    reject_negative_forward: bool = False
    reject_large_motion_body_diff: bool = False
    stabilize_suspect: bool = False


VARIANTS = (
    Variant("current"),
    Variant("suspect->forward", sidestep_suspect_mode="forward"),
    Variant(
        "suspect->forward\n+ reject negative",
        sidestep_suspect_mode="forward",
        reject_negative_forward=True,
    ),
    Variant(
        "suspect->forward\n+ reject >90deg",
        sidestep_suspect_mode="forward",
        reject_negative_forward=True,
        reject_large_motion_body_diff=True,
    ),
    Variant(
        "guarded suspect\n+ all guards",
        reject_negative_forward=True,
        reject_large_motion_body_diff=True,
        stabilize_suspect=True,
    ),
)


@dataclass
class VariantResult:
    points: list[list[float]]
    step_lengths: list[float]
    step_headings: list[pdr.StepHeading]


def _angle_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return pdr._normalize_angle(a - b)


def _is_motion_unsafe(
    step_heading: pdr.StepHeading,
    variant: Variant,
) -> bool:
    body_heading = (
        step_heading.body_heading
        if step_heading.body_heading is not None
        else step_heading.gyro_heading
    )
    if (
        variant.reject_negative_forward
        and step_heading.forward_displacement is not None
        and step_heading.forward_displacement < 0.0
    ):
        return True
    diff = _angle_diff(step_heading.motion_heading, body_heading)
    return (
        variant.reject_large_motion_body_diff
        and diff is not None
        and abs(diff) > np.deg2rad(MOTION_BODY_REJECT_DEG)
    )


def _movement_type(step_heading: pdr.StepHeading) -> str:
    return (
        step_heading.trajectory_movement_type
        if step_heading.trajectory_movement_type is not None
        else step_heading.movement_type
    )


def _guard_motion_headings(
    step_headings: list[pdr.StepHeading],
    variant: Variant,
) -> list[pdr.StepHeading]:
    guarded: list[pdr.StepHeading] = []
    for step_heading in step_headings:
        movement_type = _movement_type(step_heading)
        should_guard = movement_type == "forward" or movement_type.startswith(
            "sidestep_suspect_"
        )
        if should_guard and _is_motion_unsafe(step_heading, variant):
            guarded.append(
                step_heading._replace(
                    motion_heading=None,
                    motion_reject_reason="unsafe_motion_heading",
                )
            )
        else:
            guarded.append(step_heading)
    return guarded


def _stabilize_suspect_headings(
    step_headings: list[pdr.StepHeading],
) -> list[pdr.StepHeading]:
    stabilized = list(step_headings)
    previous_heading: float | None = None

    for index, step_heading in enumerate(step_headings):
        movement_type = _movement_type(step_heading)
        if movement_type.startswith("sidestep_suspect_"):
            body_heading = (
                step_heading.body_heading
                if step_heading.body_heading is not None
                else step_heading.gyro_heading
            )
            candidate = (
                step_heading.motion_heading
                if step_heading.motion_heading is not None
                else previous_heading
                if previous_heading is not None
                else body_heading
            )
            candidate = pdr._limit_heading_change(
                candidate,
                previous_heading,
                SUSPECT_HEADING_LIMIT_RAD,
            )
            if candidate is not None:
                stabilized[index] = step_heading._replace(
                    selected_heading=candidate,
                    source="trajectory_sidestep_suspect_guarded",
                )
                previous_heading = candidate
                continue

        if step_heading.selected_heading is not None:
            previous_heading = step_heading.selected_heading

    return stabilized


def _estimate_variant(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    peaks: np.ndarray,
    step_segments: tuple[pdr.StepSegment, ...],
    variant: Variant,
) -> VariantResult:
    device_orientation_mode = pdr._estimate_device_orientation_mode(
        df_acc,
        df_gyro,
        peaks,
        INITIAL_DIRECTION,
        step_segments,
    )
    motion_heading_correction = pdr._resolve_motion_heading_correction(
        df_acc,
        df_gyro,
        peaks,
        INITIAL_DIRECTION,
        step_segments,
        "auto",
        device_orientation_mode,
    )
    weinberg_k = compute_weinberg_k(USER_HEIGHT_M)

    raw_headings: list[pdr.StepHeading] = []
    raw_lengths: list[float] = []
    for i, peak in enumerate(peaks):
        if peak >= len(df_acc):
            continue
        step_heading = pdr.resolve_step_heading(
            peaks,
            df_gyro,
            df_acc,
            i,
            initial_direction=INITIAL_DIRECTION,
            heading_method=HEADING_METHOD,
            step_segments=step_segments,
            motion_heading_correction=motion_heading_correction,
            sidestep_lateral_ratio=SIDESTEP_LATERAL_RATIO,
            sidestep_min_lateral_displacement=SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
            device_orientation_mode=device_orientation_mode,
        )
        if step_heading.selected_heading is None:
            continue
        raw_headings.append(step_heading)
        raw_lengths.append(pdr.estimate_step_length(df_acc, int(peak), k=weinberg_k))

    smoothed = pdr._smooth_step_headings(
        raw_headings,
        SIDESTEP_SMOOTHING_METHOD,
        variant.sidestep_suspect_mode,
    )
    guarded = _guard_motion_headings(smoothed, variant)
    stabilized = pdr._stabilize_trajectory_headings(
        guarded,
        FORWARD_HEADING_SOURCE,
        "motion",
    )
    if variant.stabilize_suspect:
        stabilized = _stabilize_suspect_headings(stabilized)

    points: list[list[float]] = [[0.0, 0.0]]
    step_lengths: list[float] = []
    step_headings: list[pdr.StepHeading] = []
    previous_heading: float | None = None
    for step_heading, step_length in zip(stabilized, raw_lengths, strict=True):
        step_motion = pdr.estimate_step_motion(
            step_heading,
            step_length,
            previous_heading,
            FORWARD_HEADING_SOURCE,
            "motion",
            variant.sidestep_suspect_mode,
        )
        if step_motion is None:
            continue
        source = (
            step_heading.source
            if step_heading.source.startswith("trajectory_")
            else "state_motion"
        )
        step_heading = step_heading._replace(
            selected_heading=step_motion.heading,
            source=source,
            step_length_scale=step_motion.length_scale,
            trajectory_movement_type=step_motion.movement_type,
            forward_heading_source=FORWARD_HEADING_SOURCE,
        )
        step_lengths.append(step_motion.length)
        step_headings.append(step_heading)
        previous_heading = step_motion.heading
        points.append(
            [
                points[-1][0] + step_motion.length * float(np.cos(step_motion.heading)),
                points[-1][1] + step_motion.length * float(np.sin(step_motion.heading)),
            ]
        )

    return VariantResult(points, step_lengths, step_headings)


def _selected_delta_stats(step_headings: list[pdr.StepHeading]) -> tuple[float, float]:
    headings = np.array(
        [
            np.nan if heading.selected_heading is None else heading.selected_heading
            for heading in step_headings
        ],
        dtype=float,
    )
    deltas = np.abs(
        np.array(
            [
                np.nan,
                *[
                    pdr._normalize_angle(headings[i] - headings[i - 1])
                    for i in range(1, len(headings))
                ],
            ],
            dtype=float,
        )
    )
    return float(np.nanpercentile(np.rad2deg(deltas), 90)), float(
        np.nanmax(np.rad2deg(deltas))
    )


def _plot_result(
    ax: plt.Axes,
    dataset_name: str,
    variant: Variant,
    result: VariantResult,
) -> None:
    points = np.array(result.points)
    ax.plot(points[:, 0], points[:, 1], marker="o", markersize=2, linewidth=1.3)
    ax.scatter(points[0, 0], points[0, 1], color="green", s=28, label="start")
    ax.scatter(points[-1, 0], points[-1, 1], color="red", s=28, label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_title(f"{dataset_name}\n{variant.name}", fontsize=10)

    move = pd.Series(
        [
            heading.trajectory_movement_type
            if heading.trajectory_movement_type is not None
            else heading.movement_type
            for heading in result.step_headings
        ]
    )
    p90, max_delta = _selected_delta_stats(result.step_headings)
    neg_forward = sum(
        1
        for heading in result.step_headings
        if heading.forward_displacement is not None and heading.forward_displacement < 0
    )
    suspect = int(move.astype(str).str.startswith("sidestep_suspect").sum())
    ax.text(
        0.02,
        0.02,
        f"end=({points[-1, 0]:.1f},{points[-1, 1]:.1f})\n"
        f"delta p90/max={p90:.1f}/{max_delta:.1f}deg\n"
        f"neg={neg_forward} suspect={suspect}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )


def main() -> None:
    results: dict[tuple[str, str], VariantResult] = {}
    for dataset in DATASETS:
        df_acc_raw, df_gyro_raw = pdr.load_sensor_data(dataset)
        df_acc, df_gyro = pdr.process_sensor_data(
            df_acc_raw,
            df_gyro_raw,
        )
        step_detection = pdr.detect_step_result(df_acc)
        for variant in VARIANTS:
            results[(dataset.name, variant.name)] = _estimate_variant(
                df_acc,
                df_gyro,
                step_detection.peaks,
                step_detection.segments,
                variant,
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(DATASETS),
        len(VARIANTS),
        figsize=(4.2 * len(VARIANTS), 4.2 * len(DATASETS)),
        squeeze=False,
    )
    for row, dataset in enumerate(DATASETS):
        for col, variant in enumerate(VARIANTS):
            _plot_result(
                axes[row][col],
                dataset.name.replace(
                    "ryuki_1turn_rightsidestep_3turn_leftsidestep",
                    "ryuki",
                ),
                variant,
                results[(dataset.name, variant.name)],
            )
    fig.suptitle("Ryuki heading guard comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Saved comparison plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
