from __future__ import annotations

from screen_agent_mvp.reporting import render_markdown_report
from screen_agent_mvp.schema import EvidenceItem, FrameObservation, Report, VideoMetadata


def test_render_markdown_report_contains_core_sections() -> None:
    metadata = VideoMetadata("input.mp4", 3.0, 30.0, 90, 1280, 720)
    observation = FrameObservation(1, 0.0, "frame.jpg", "interval", [], [])
    report = Report(
        title="Checkout failure",
        outcome="failure_suspected",
        summary="Payment did not complete.",
        repro_steps=["Open checkout", "Click Pay"],
        expected="Order completes.",
        actual="Validation error appears.",
        evidence=[EvidenceItem(1.0, "Error", "ZIP required", "frame.jpg")],
        uncertainty=[],
        warnings=[],
        preset="bug-report",
    )

    markdown = render_markdown_report(report, metadata, [observation])

    assert "# Checkout failure" in markdown
    assert "## Repro Steps" in markdown
    assert "ZIP required" in markdown
    assert "Preset: `bug-report`" in markdown
