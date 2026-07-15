from __future__ import annotations

import time
import uuid
from pathlib import Path

from .config import AnalysisConfig
from .engines import build_engines
from .reporting import write_report_bundle
from .schema import EvidenceItem, FrameObservation, Report, TimelineEvent, VideoMetadata
from .video import sample_video_frames, write_evidence_clip


def default_run_dir(root: str | Path = "runs") -> Path:
    return Path(root) / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])


def analyze_video(
    video_path: str | Path,
    output_dir: str | Path | None,
    config: AnalysisConfig,
) -> dict[str, object]:
    run_dir = Path(output_dir) if output_dir else default_run_dir()
    keyframes_dir = run_dir / "keyframes"
    metadata, frames = sample_video_frames(
        path=video_path,
        output_dir=keyframes_dir,
        sample_fps=config.sample_fps,
        max_frames=config.max_frames,
        scene_change_threshold=config.scene_change_threshold,
    )
    ocr_engine, ui_engine, reasoning_engine = build_engines(config)
    observations: list[FrameObservation] = []
    for frame in frames:
        ocr_blocks, ocr_warnings = ocr_engine.detect(frame.path)
        ui_elements, ui_warnings = ui_engine.parse(frame.path, ocr_blocks)
        observations.append(
            FrameObservation(
                index=frame.index,
                timestamp_seconds=frame.timestamp_seconds,
                frame_path=frame.path,
                reason=frame.reason,
                ocr_blocks=ocr_blocks,
                ui_elements=ui_elements,
                warnings=ocr_warnings + ui_warnings,
            )
        )
    events = build_timeline_events(observations)
    report = reasoning_engine.build_report(_select_key_observations(observations, config))
    report = _attach_preset_and_clips(
        report=report,
        preset=config.preset,
        video_path=video_path,
        run_dir=run_dir,
        metadata=metadata,
    )
    artifacts = write_report_bundle(run_dir, metadata, observations, events, report)
    return {
        "run_dir": str(run_dir),
        "metadata": metadata,
        "observations": observations,
        "events": events,
        "report": report,
        "artifacts": artifacts,
    }


def build_timeline_events(observations: list[FrameObservation]) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    previous_text: set[str] = set()
    error_terms = ("error", "failed", "invalid", "required", "denied", "try again", "disabled")
    for observation in observations:
        texts = {block.text.strip() for block in observation.ocr_blocks if block.text.strip()}
        new_text = sorted(texts - previous_text)
        joined_new_text = " ".join(new_text)
        if any(term in joined_new_text.lower() for term in error_terms):
            events.append(
                TimelineEvent(
                    timestamp_seconds=observation.timestamp_seconds,
                    kind="possible-error",
                    title="Possible error state",
                    detail=joined_new_text[:300],
                    frame_path=observation.frame_path,
                    confidence=0.75,
                )
            )
        elif observation.reason == "scene_change":
            events.append(
                TimelineEvent(
                    timestamp_seconds=observation.timestamp_seconds,
                    kind="scene-change",
                    title="UI scene changed",
                    detail=joined_new_text[:300] or "Visual scene-change sampler selected this frame.",
                    frame_path=observation.frame_path,
                    confidence=0.45,
                )
            )
        elif new_text and not events:
            events.append(
                TimelineEvent(
                    timestamp_seconds=observation.timestamp_seconds,
                    kind="initial-state",
                    title="Initial visible state",
                    detail=" | ".join(new_text[:8]),
                    frame_path=observation.frame_path,
                    confidence=0.5,
                )
            )
        previous_text = texts
    if observations and not any(event.timestamp_seconds == observations[-1].timestamp_seconds for event in events):
        events.append(
            TimelineEvent(
                timestamp_seconds=observations[-1].timestamp_seconds,
                kind="final-state",
                title="Final sampled state",
                detail="Last sampled frame in the selected recording window.",
                frame_path=observations[-1].frame_path,
                confidence=0.5,
            )
        )
    return events[:20]


def _select_key_observations(
    observations: list[FrameObservation],
    config: AnalysisConfig,
) -> list[FrameObservation]:
    if len(observations) <= config.max_keyframes_for_vlm:
        return observations
    selected: list[FrameObservation] = []
    selected.append(observations[0])
    error_terms = ("error", "failed", "invalid", "required", "denied", "try again")
    for observation in observations:
        text = " ".join(block.text.lower() for block in observation.ocr_blocks)
        if any(term in text for term in error_terms) or observation.reason == "scene_change":
            selected.append(observation)
        if len(selected) >= config.max_keyframes_for_vlm - 1:
            break
    selected.append(observations[-1])
    deduped: dict[int, FrameObservation] = {item.index: item for item in selected}
    return list(deduped.values())[: config.max_keyframes_for_vlm]


def _attach_preset_and_clips(
    report: Report,
    preset: str,
    video_path: str | Path,
    run_dir: Path,
    metadata: VideoMetadata,
) -> Report:
    clips_dir = run_dir / "evidence_clips"
    evidence: list[EvidenceItem] = []
    for index, item in enumerate(report.evidence, start=1):
        clip_path: str | None = None
        try:
            clip = write_evidence_clip(
                video_path=video_path,
                timestamp_seconds=item.timestamp_seconds,
                output_path=clips_dir / f"evidence-{index:02d}-{item.timestamp_seconds:.3f}s.mp4",
                metadata=metadata,
            )
            clip_path = str(clip)
        except Exception:
            clip_path = None
        evidence.append(
            EvidenceItem(
                timestamp_seconds=item.timestamp_seconds,
                title=item.title,
                detail=item.detail,
                frame_path=item.frame_path,
                clip_path=clip_path,
            )
        )
    return Report(
        title=report.title,
        outcome=report.outcome,
        summary=report.summary,
        repro_steps=report.repro_steps,
        expected=report.expected,
        actual=report.actual,
        evidence=evidence,
        uncertainty=report.uncertainty,
        warnings=report.warnings,
        preset=preset,
    )


def summarize_result(result: dict[str, object]) -> str:
    metadata = result["metadata"]
    report = result["report"]
    if not isinstance(metadata, VideoMetadata) or not isinstance(report, Report):
        return "Analysis complete."
    return (
        f"{report.title}\n"
        f"Outcome: {report.outcome}\n"
        f"Video: {metadata.duration_seconds:.2f}s, {metadata.width}x{metadata.height}\n"
        f"Artifacts: {result['artifacts']}"
    )
