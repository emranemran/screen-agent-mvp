from __future__ import annotations

import json
from pathlib import Path

from screen_agent_mvp.qa import answer_run_question, render_answer_markdown


def test_answer_run_question_uses_report_and_evidence(tmp_path: Path) -> None:
    _write_run(tmp_path)

    result = answer_run_question(tmp_path, "why did checkout fail?")

    assert "ZIP code" in result.answer
    assert result.evidence
    assert result.evidence[0]["timestamp_seconds"] == 4.0
    assert "Frame:" in render_answer_markdown(result)


def test_answer_run_question_handles_timestamp_question(tmp_path: Path) -> None:
    _write_run(tmp_path)

    result = answer_run_question(tmp_path, "what timestamp shows the error?")

    assert "4.00s" in result.answer


def _write_run(path: Path) -> None:
    (path / "summary.json").write_text(
        json.dumps(
            {
                "video": {"path": "demo.mp4"},
                "report": {
                    "title": "Payment Form Validation Bug",
                    "outcome": "vlm_generated",
                    "summary": "Checkout failed because ZIP code is required.",
                    "expected": "Order should proceed after valid input.",
                    "actual": "The Pay button remains disabled.",
                    "repro_steps": ["Open checkout", "Click Pay"],
                    "evidence": [
                        {
                            "timestamp_seconds": 4.0,
                            "title": "Frame 2",
                            "detail": "Error: ZIP code is required",
                            "frame_path": "keyframes/frame-0002.jpg",
                            "clip_path": "evidence_clips/evidence-01.mp4",
                        }
                    ],
                    "uncertainty": [],
                    "warnings": [],
                    "preset": "bug-report",
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "timeline.json").write_text(
        json.dumps(
            {
                "video": {"path": "demo.mp4"},
                "events": [
                    {
                        "timestamp_seconds": 4.0,
                        "kind": "possible-error",
                        "title": "Possible error state",
                        "detail": "Error: ZIP code is required",
                        "frame_path": "keyframes/frame-0002.jpg",
                        "confidence": 0.75,
                    }
                ],
                "frames": [
                    {
                        "timestamp_seconds": 4.0,
                        "frame_path": "keyframes/frame-0002.jpg",
                        "ocr_blocks": [{"text": "Error: ZIP code is required"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

