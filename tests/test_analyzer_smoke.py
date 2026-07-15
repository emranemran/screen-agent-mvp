from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from screen_agent_mvp.analyzer import analyze_video
from screen_agent_mvp.config import AnalysisConfig


def test_analyze_video_writes_report_bundle(tmp_path: Path) -> None:
    video_path = tmp_path / "screen.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (320, 180),
    )
    assert writer.isOpened()
    try:
        for index in range(6):
            frame = np.full((180, 320, 3), 255, dtype=np.uint8)
            cv2.putText(
                frame,
                "ZIP required" if index >= 3 else "Checkout",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()

    result = analyze_video(
        video_path,
        tmp_path / "run",
        AnalysisConfig(
            sample_fps=1.0,
            max_frames=5,
            strict_models=False,
            force_fallback_models=True,
        ),
    )

    artifacts = result["artifacts"]
    assert Path(artifacts["report"]).exists()
    assert Path(artifacts["timeline"]).exists()
    assert Path(artifacts["summary"]).exists()
    assert result["events"]
    assert "No clear failure detected" in Path(artifacts["report"]).read_text(encoding="utf-8")
    assert "Clip:" in Path(artifacts["report"]).read_text(encoding="utf-8")
