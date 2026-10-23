"""粒子動画の形式選択と保存失敗時の出力保護を検証する。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from rikka.plot.lib import animation


@pytest.mark.parametrize(
    ("filename", "failures", "expected"),
    [
        ("clip.mp4", (), [".mp4"]),
        ("clip.webm", (), [".webm"]),
        ("clip.mp4", (".mp4",), [".mp4", ".webm"]),
        ("clip.webm", (".webm",), [".webm", ".mp4"]),
        ("clip.mp4", (".mp4", ".webm"), [".mp4", ".webm", ".gif"]),
    ],
)
def test_animation_prefers_requested_format_and_cleans_partial_files(
    tmp_path, monkeypatch, filename, failures, expected
) -> None:
    """指定形式を先に試し、失敗した新規ファイルを残さず保存する。"""
    map_path = tmp_path / "map.png"
    plt.imsave(map_path, np.ones((20, 20)), cmap="gray")
    monkeypatch.setattr(animation, "configure_japanese_font", lambda: None)
    calls = []

    def fake_save(self, filename, writer):
        self._draw_was_started = True
        path = Path(filename)
        calls.append(path.suffix)
        if path.suffix == ".mp4":
            assert writer.codec == "libx264"
        elif path.suffix == ".webm":
            assert writer.codec == "libvpx-vp9"
        path.write_bytes(b"partial" if path.suffix in failures else b"complete")
        if path.suffix in failures:
            raise RuntimeError("エンコーダーを利用できません")

    monkeypatch.setattr(FuncAnimation, "save", fake_save)
    destination = tmp_path / filename
    destination.write_bytes(b"existing")
    figures_before = plt.get_fignums()
    result = animation.save_particle_animation(
        np.zeros((2, 3, 2)),
        [[0.0, 0.0], [0.1, 0.1]],
        gx_mean=0.0,
        gz_mean=1.0,
        floormap_path=map_path,
        origin_px=(10, 10),
        scale=1.0,
        output_path=destination,
    )

    assert calls == expected
    assert result == destination.with_suffix(expected[-1])
    assert result.read_bytes() == b"complete"
    assert not list(tmp_path.glob("*.tmp.*"))
    assert plt.get_fignums() == figures_before
    if destination.suffix in failures:
        assert destination.read_bytes() == b"existing"


def test_animation_closes_figure_when_all_encoders_fail(tmp_path, monkeypatch) -> None:
    """全形式で失敗しても一時ファイルとfigureを残さない。"""
    map_path = tmp_path / "map.png"
    plt.imsave(map_path, np.ones((20, 20)), cmap="gray")
    monkeypatch.setattr(animation, "configure_japanese_font", lambda: None)

    def fail(self, filename, writer):
        self._draw_was_started = True
        Path(filename).write_bytes(b"partial")
        raise RuntimeError("保存失敗")

    monkeypatch.setattr(FuncAnimation, "save", fail)
    figures_before = plt.get_fignums()
    with pytest.raises(RuntimeError, match="保存失敗"):
        animation.save_particle_animation(
            np.zeros((2, 3, 2)),
            [[0.0, 0.0], [0.1, 0.1]],
            gx_mean=0.0,
            gz_mean=1.0,
            floormap_path=map_path,
            output_path=tmp_path / "clip.mp4",
        )
    assert not list(tmp_path.glob("clip*"))
    assert plt.get_fignums() == figures_before
