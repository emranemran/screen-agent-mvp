from __future__ import annotations

from screen_agent_mvp.video import build_base_sample_times


def test_build_base_sample_times_caps_frames() -> None:
    assert build_base_sample_times(10.0, sample_fps=1.0, max_frames=3) == [0.0, 1.0, 2.0]


def test_build_base_sample_times_includes_end_when_room() -> None:
    assert build_base_sample_times(2.2, sample_fps=1.0, max_frames=5)[-1] == 2.1


def test_build_base_sample_times_does_not_exceed_cap_for_end_frame() -> None:
    assert len(build_base_sample_times(10.0, sample_fps=1.0, max_frames=3)) == 3
