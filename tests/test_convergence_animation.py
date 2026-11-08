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


@pytest.mark.parametrize("converged", [True, False])
@pytest.mark.parametrize("holds", [None, (1, 4, 2), (0, 0, 0)])
def test_story_frames_and_phases(
    synthetic_story, tmp_path, monkeypatch, converged, holds
):
    """通常歩と各holdの加算、軌跡の伸長、色分け、収束表示を確認する。"""
    particle, map_path = synthetic_story
    if not converged:
        particle = replace(particle, convergence=None)
    fps = 2
    initial, convergence, final = (4, 6, 6) if holds is None else holds
    observed = []

    def inspect_save(self, filename, writer):
        self._draw_was_started = True
        frames = list(self.new_frame_seq())
        assert len(frames) == 3 + initial + (convergence if converged else 0) + final
        for frame in frames:
            self._func(frame)
            map_ax, cluster_ax, spread_ax, text_ax = self._fig.axes
            index, phase, _ = frame
            text = text_ax.texts[0].get_text()
            observed.append((phase, text))
            assert len(map_ax.lines[0].get_xdata()) == index + 2
            assert len(map_ax.collections[0].get_offsets()) == 1
            assert len(map_ax.collections[1].get_offsets()) == 2
            assert cluster_ax.lines[1].get_xdata() == [index + 1] * 2
            assert spread_ax.lines[1].get_xdata() == [index + 1] * 2
            assert len(map_ax.patches) == int(converged and index >= 1)
            if phase == "initial":
                assert "初期状態:" in text
            elif phase == "convergence" or (converged and index == 1):
                assert "収束 K=2" in text
            elif phase == "final":
                assert "最終軌跡: 全3歩" in text
            elif converged and index == 2:
                assert "追従中" in text
            if not converged and index == 2:
                assert "未収束のまま終了" in text
        Path(filename).write_bytes(b"complete")

    monkeypatch.setattr(FuncAnimation, "save", inspect_save)
    arguments = (
        {}
        if holds is None
        else dict(
            initial_hold_frames=initial,
            convergence_hold_frames=convergence,
            final_hold_frames=final,
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
    )
    assert path.suffix == ".gif"
    with Image.open(path) as image:
        assert image.n_frames == 3
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
