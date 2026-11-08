"""合成地図と診断列で収束説明動画の時間構成と描画を検証する。"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FFMpegWriter, FuncAnimation
from PIL import Image

from rikka.common.lib.models import (
    ParticleClusterSummary,
    ParticleConvergenceDiagnostic,
    ParticleFilterResult,
)
from rikka.common.settings import OutputSettings, ParticleSettings
from rikka.plot import pipeline
from rikka.plot.lib import convergence_animation as story


@pytest.fixture
def synthetic_story(tmp_path):
    """初期配置と3歩、2歩目の収束候補を持つ最小履歴を作る。"""
    map_path = tmp_path / "map.png"
    plt.imsave(map_path, np.ones((20, 20)), cmap="gray", vmin=0, vmax=1)
    diagnostics = tuple(
        ParticleConvergenceDiagnostic(
            step_number=step,
            step_time=float(step),
            cluster_count=4 - step,
            dominant_weight_ratio=0.8,
            dominant_spread_rms_m=0.2,
            dominant_centroid_x_m=float(step),
            dominant_centroid_y_m=1.0,
            position_spread_rms_m=2.0,
            surviving_particle_count=3,
            top_clusters=(ParticleClusterSummary(float(step), 1.0, 0.8, 0.2),),
            convergence_step=2 if step >= 2 else None,
        )
        for step in range(1, 4)
    )
    particle = ParticleFilterResult(
        trajectory=[[float(step), 1.0] for step in range(4)],
        all_particles=[
            np.array([[float(step), 1.0], [float(step) + 0.1, 1.0], [10.0, 10.0]])
            for step in range(4)
        ],
        convergence_diagnostics=diagnostics,
        convergence=diagnostics[1],
    )
    return particle, map_path


@pytest.mark.parametrize("stride", [1, 2, 5])
@pytest.mark.parametrize("converged", [True, False])
@pytest.mark.parametrize("holds", [None, (1, 4, 2), (0, 0, 0)])
def test_story_frames_and_phases(
    synthetic_story, tmp_path, monkeypatch, converged, holds, stride
):
    """通常歩と各holdの加算、軌跡の伸長、色分け、収束表示を確認する。"""
    particle, map_path = synthetic_story
    if not converged:
        particle = replace(particle, convergence=None)
    fps = 2
    initial, convergence, final = (4, 6, 6) if holds is None else holds
    observed = []
    rewind_indices = list(range(2, -1, -stride))
    if rewind_indices[-1] != 0:
        rewind_indices.append(0)
    rewind_hold, rewind_final = (4, 6) if holds is None else (1, 2)
    rewind_lengths = []

    def inspect_save(self, filename, writer):
        self._draw_was_started = True
        frames = list(self.new_frame_seq())
        assert len(frames) == (
            3
            + initial
            + (convergence if converged else 0)
            + final
            + rewind_hold
            + len(rewind_indices)
            + rewind_final
        )
        assert [
            index for index, phase, _ in frames if phase == "rewind"
        ] == rewind_indices
        thumbnails = None
        for frame in frames:
            self._func(frame)
            map_ax, *thumbnail_axes, text_ax = self._fig.axes
            index, phase, _ = frame
            text = text_ax.texts[0].get_text()
            observed.append((phase, text))
            current_thumbnails = [
                (
                    ax.get_title(),
                    ax.collections[0].get_offsets().copy(),
                    ax.images[0].get_array().copy(),
                )
                for ax in thumbnail_axes
            ]
            if thumbnails is None:
                thumbnails = current_thumbnails
            for ax, (caption, points, image), previous in zip(
                thumbnail_axes, current_thumbnails, thumbnails, strict=True
            ):
                assert not len(ax.get_xticks()) and not len(ax.get_yticks())
                assert not ax.lines
                assert caption == previous[0]
                np.testing.assert_array_equal(points, previous[1])
                np.testing.assert_array_equal(image, previous[2])
            assert thumbnail_axes[0].get_title() == "1歩目: 地図全体へ一様配置"
            assert "1歩目: 候補が最も分かれた時点（クラスタ3個）" == (
                thumbnail_axes[1].get_title()
            )
            assert thumbnail_axes[2].get_title() == (
                "2歩目: 収束" if converged else "最終歩: 未収束のまま終了"
            )
            np.testing.assert_array_equal(
                thumbnail_axes[2].collections[0].get_offsets()[:, 0],
                particle.all_particles[2 if converged else 3][:, 0],
            )
            if phase.startswith("rewind"):
                assert len(map_ax.lines[0].get_xdata()) == 3 - index
                np.testing.assert_array_equal(
                    map_ax.lines[0].get_xdata(),
                    np.asarray(particle.trajectory)[index + 1 :, 0],
                )
                assert len(map_ax.collections[0].get_offsets()) == 3
                assert map_ax.collections[1].get_label() == "この歩のときの祖先"
                np.testing.assert_array_equal(
                    map_ax.collections[1].get_offsets()[:, 0], [index + 1]
                )
                assert len(map_ax.patches) == int(converged)
                if phase == "rewind":
                    rewind_lengths.append(len(map_ax.lines[0].get_xdata()))
                if phase == "rewind_start" or index == 2:
                    assert text == "巻き戻し: 最後に残った粒子の親を1歩ずつ遡る"
                elif index == 0:
                    assert text == (
                        "収束前の区間は、逆方向PFではなく生き残った粒子の系譜"
                    )
                else:
                    assert text == "2歩目の祖先 / ここから先は実際に粒子が通った経路"
                continue
            assert len(map_ax.lines[0].get_xdata()) == index + 2
            assert len(map_ax.collections[0].get_offsets()) == 1
            assert len(map_ax.collections[1].get_offsets()) == 2
            assert len(map_ax.patches) == int(converged and index >= 1)
            if phase == "initial":
                assert "初期状態:" in text
            elif phase == "final":
                assert "最終軌跡: 全3歩" in text
            elif phase == "convergence" or (converged and index == 1):
                assert "収束 K=2" in text
            elif converged and index == 2:
                assert "追従中 / 収束位置から" in text
            else:
                assert "探索中（前向き1パス）/ 候補クラスタ" in text
            if not converged and index == 2:
                assert "未収束のまま終了" in text
        assert all(
            a < b for a, b in zip(rewind_lengths, rewind_lengths[1:], strict=False)
        )
        Path(filename).write_bytes(b"complete")

    monkeypatch.setattr(FuncAnimation, "save", inspect_save)
    arguments = (
        {}
        if holds is None
        else dict(
            initial_hold_frames=initial,
            convergence_hold_frames=convergence,
            final_hold_frames=final,
            rewind_hold_frames=rewind_hold,
            rewind_final_hold_frames=rewind_final,
        )
    )
    before = plt.get_fignums()
    path = story.save_convergence_animation(
        particle,
        0.0,
        1.0,
        floormap_path=map_path,
        origin_px=(0, 0),
        scale=1.0,
        output_path=tmp_path / "convergence_story.mp4",
        fps=fps,
        rewind_stride=stride,
        **arguments,
    )
    assert path.read_bytes() == b"complete"
    assert plt.get_fignums() == before
    assert sum(phase == "convergence" for phase, _ in observed) == (
        convergence if converged else 0
    )


def test_story_falls_back_to_real_gif(synthetic_story, tmp_path, monkeypatch):
    """ffmpegが利用できなくてもPillowで実際にGIFを書き出す。"""
    particle, map_path = synthetic_story

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("ffmpeg がありません")

    monkeypatch.setattr(FFMpegWriter, "setup", unavailable)
    path = story.save_convergence_animation(
        particle,
        0.0,
        1.0,
        floormap_path=map_path,
        origin_px=(0, 0),
        scale=1.0,
        output_path=tmp_path / "convergence_story.mp4",
        initial_hold_frames=0,
        convergence_hold_frames=0,
        final_hold_frames=0,
        rewind_hold_frames=0,
        rewind_final_hold_frames=0,
    )
    assert path.suffix == ".gif"
    with Image.open(path) as image:
        assert image.n_frames == 6
    assert not list(tmp_path.glob("*.tmp.*"))


@pytest.mark.parametrize(
    ("mode", "enabled", "has_diagnostics", "expected"),
    [
        ("uniform", True, True, ["particle", "story"]),
        ("origin", True, True, ["particle"]),
        ("uniform", False, True, []),
        ("uniform", True, False, ["particle"]),
    ],
)
def test_story_pipeline_conditions(
    synthetic_story, tmp_path, monkeypatch, mode, enabled, has_diagnostics, expected
):
    """保存条件を満たした場合だけ、既存動画の直後に説明動画を保存する。"""
    particle, map_path = synthetic_story
    if not has_diagnostics:
        particle = replace(particle, convergence_diagnostics=(), convergence=None)
    calls = []
    monkeypatch.setattr(
        pipeline, "save_particle_animation", lambda *a, **kw: calls.append("particle")
    )

    def save_story(result, **kwargs):
        assert result is particle
        assert kwargs["output_path"] == tmp_path / "convergence_story.mp4"
        assert kwargs["cell_m"] == 2.0
        calls.append("story")

    monkeypatch.setattr(pipeline, "save_convergence_animation", save_story)
    result = SimpleNamespace(
        particle=particle,
        prepared=SimpleNamespace(gx_mean=0.0, gz_mean=1.0),
        trajectory=particle.trajectory,
        landmark=None,
    )
    pipeline._render_particle_artifacts(
        result,
        ParticleSettings(
            init_mode=mode, floormap_path=map_path, convergence_cell_m=2.0
        ),
        OutputSettings(plot=False, save_animation=enabled),
        tmp_path,
    )
    assert calls == expected


@pytest.mark.parametrize("rewind_final", [0, 1])
@pytest.mark.parametrize("single_step", [False, True])
def test_story_thumbnail_peak_and_original_steps(
    synthetic_story, tmp_path, monkeypatch, single_step, rewind_final
):
    """最大数が同じなら早い診断を選び、欠番と1歩だけの履歴にも対応する。"""
    particle, map_path = synthetic_story
    count = 1 if single_step else 3
    diagnostics = tuple(
        replace(item, step_number=step, cluster_count=clusters)
        for item, step, clusters in zip(
            particle.convergence_diagnostics[:count],
            (1, 4, 7)[:count],
            (2, 5, 5)[:count],
            strict=True,
        )
    )
    particle = replace(
        particle,
        convergence=None,
        convergence_diagnostics=diagnostics,
        trajectory=particle.trajectory[: count + 1],
        all_particles=particle.all_particles[: count + 1],
    )

    def inspect_save(self, filename, writer):
        self._draw_was_started = True
        frames = list(self.new_frame_seq())
        assert len(frames) == count * 2 + rewind_final
        assert not any(phase == "convergence" for _, phase, _ in frames)
        for frame in frames:
            self._func(frame)
        map_ax, _, peak_ax, _, text_ax = self._fig.axes
        peak_index = 0 if single_step else 1
        assert peak_ax.get_title().startswith(
            f"{diagnostics[peak_index].step_number}歩目: 候補が最も分かれた時点"
        )
        np.testing.assert_array_equal(
            peak_ax.collections[0].get_offsets()[:, 0],
            particle.all_particles[peak_index + 1][:, 0],
        )
        assert len(map_ax.lines[0].get_xdata()) == count
        assert text_ax.texts[0].get_text() == (
            "収束前の区間は、逆方向PFではなく生き残った粒子の系譜"
        )
        Path(filename).write_bytes(b"complete")

    monkeypatch.setattr(FuncAnimation, "save", inspect_save)
    story.save_convergence_animation(
        particle,
        0.0,
        1.0,
        floormap_path=map_path,
        origin_px=(0, 0),
        scale=1.0,
        output_path=tmp_path / "story.mp4",
        initial_hold_frames=0,
        convergence_hold_frames=5,
        final_hold_frames=0,
        rewind_hold_frames=0,
        rewind_final_hold_frames=rewind_final,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"rewind_stride": 0},
        {"rewind_stride": -1},
        {"rewind_hold_frames": -1},
        {"rewind_final_hold_frames": -1},
    ],
)
def test_story_rejects_invalid_rewind_arguments(synthetic_story, arguments):
    """巻き戻しの間隔は正、静止フレーム数は非負に制限する。"""
    particle, map_path = synthetic_story
    with pytest.raises(ValueError, match="rewind_stride"):
        story.save_convergence_animation(
            particle, 0.0, 1.0, floormap_path=map_path, **arguments
        )
