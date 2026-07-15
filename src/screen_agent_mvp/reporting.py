from __future__ import annotations

import json
from pathlib import Path

from .schema import FrameObservation, Report, TimelineEvent, VideoMetadata, to_plain


def write_json(path: str | Path, payload) -> None:
    Path(path).write_text(json.dumps(to_plain(payload), indent=2), encoding="utf-8")


def render_markdown_report(report: Report, metadata: VideoMetadata, observations: list[FrameObservation]) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"**Outcome:** `{report.outcome}`",
        "",
        "## Summary",
        report.summary or "No summary generated.",
        "",
        "## Video",
        f"- Source: `{metadata.path}`",
        f"- Preset: `{report.preset}`",
        f"- Duration: {metadata.duration_seconds:.2f}s",
        f"- Resolution: {metadata.width}x{metadata.height}",
        f"- Sampled frames: {len(observations)}",
        "",
        "## Repro Steps",
    ]
    if report.repro_steps:
        lines.extend(f"{index}. {step}" for index, step in enumerate(report.repro_steps, start=1))
    else:
        lines.append("No explicit repro steps generated.")
    lines.extend(["", "## Expected", report.expected, "", "## Actual", report.actual, "", "## Evidence"])
    for item in report.evidence:
        lines.extend(
            [
                f"- **{item.timestamp_seconds:.2f}s - {item.title}**",
                f"  - {item.detail}",
                f"  - Frame: `{item.frame_path}`",
                *([f"  - Clip: `{item.clip_path}`"] if item.clip_path else []),
            ]
        )
    if not report.evidence:
        lines.append("No evidence items generated.")
    lines.extend(["", "## Uncertainty"])
    lines.extend(f"- {item}" for item in report.uncertainty) if report.uncertainty else lines.append("- None")
    if report.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in sorted(set(report.warnings)))
    return "\n".join(lines) + "\n"


def write_report_bundle(
    output_dir: str | Path,
    metadata: VideoMetadata,
    observations: list[FrameObservation],
    events: list[TimelineEvent],
    report: Report,
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / "report.md"
    timeline_path = target / "timeline.json"
    summary_path = target / "summary.json"
    report_path.write_text(render_markdown_report(report, metadata, observations), encoding="utf-8")
    write_json(timeline_path, {"video": metadata, "events": events, "frames": observations})
    write_json(summary_path, {"video": metadata, "report": report})
    return {
        "report": str(report_path),
        "timeline": str(timeline_path),
        "summary": str(summary_path),
    }
