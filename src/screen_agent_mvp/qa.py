from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunAnswer:
    question: str
    answer: str
    evidence: list[dict[str, Any]]


def answer_run_question(run_dir: str | Path, question: str) -> RunAnswer:
    run_path = Path(run_dir)
    if not question.strip():
        raise ValueError("Question must not be empty.")
    summary_path = run_path / "summary.json"
    timeline_path = run_path / "timeline.json"
    if not summary_path.exists() or not timeline_path.exists():
        raise ValueError(f"Run directory must contain summary.json and timeline.json: {run_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    report = summary.get("report", {})
    events = timeline.get("events", [])
    frames = timeline.get("frames", [])
    terms = _question_terms(question)
    ranked_events = _rank_events(events, terms)
    ranked_frames = _rank_frames(frames, terms)
    evidence = _build_evidence(report, ranked_events, ranked_frames)

    answer = _compose_answer(question, report, evidence)
    return RunAnswer(question=question, answer=answer, evidence=evidence)


def render_answer_markdown(result: RunAnswer) -> str:
    lines = ["## Answer", result.answer, "", "## Evidence"]
    if not result.evidence:
        lines.append("- No matching evidence found in this run.")
    for item in result.evidence:
        lines.extend(
            [
                f"- **{item['timestamp_seconds']:.2f}s - {item['title']}**",
                f"  - {item['detail']}",
                f"  - Frame: `{item['frame_path']}`",
            ]
        )
        if item.get("clip_path"):
            lines.append(f"  - Clip: `{item['clip_path']}`")
    return "\n".join(lines) + "\n"


def _compose_answer(question: str, report: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    lower = question.lower()
    title = str(report.get("title") or "this run")
    summary = str(report.get("summary") or "").strip()
    actual = str(report.get("actual") or "").strip()
    expected = str(report.get("expected") or "").strip()
    repro = [str(step) for step in report.get("repro_steps") or [] if str(step).strip()]
    timestamps = ", ".join(f"{item['timestamp_seconds']:.2f}s" for item in evidence[:4])

    if "timestamp" in lower or "when" in lower or "where" in lower:
        return (
            f"The most relevant evidence for **{title}** appears at {timestamps}."
            if timestamps
            else f"I could not find a matching timestamp for **{title}** in this run."
        )
    if "step" in lower or "repro" in lower or "reproduce" in lower:
        if repro:
            return "Repro steps from the run:\n" + "\n".join(
                f"{index}. {step}" for index, step in enumerate(repro, start=1)
            )
        return "No explicit repro steps were generated for this run."
    if "expected" in lower:
        return expected or "No expected behavior was recorded for this run."
    if "actual" in lower:
        return actual or "No actual behavior was recorded for this run."
    if "why" in lower or "fail" in lower or "error" in lower:
        parts = [summary, actual]
        answer = " ".join(part for part in parts if part)
        return answer or f"The run is titled **{title}**, but no failure explanation was recorded."
    return summary or actual or f"The run is titled **{title}**."


def _build_evidence(
    report: dict[str, Any],
    ranked_events: list[dict[str, Any]],
    ranked_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    report_evidence = report.get("evidence") or []
    for item in report_evidence[:3]:
        evidence.append(
            {
                "timestamp_seconds": float(item.get("timestamp_seconds") or 0),
                "title": str(item.get("title") or "Report evidence"),
                "detail": str(item.get("detail") or ""),
                "frame_path": str(item.get("frame_path") or ""),
                "clip_path": item.get("clip_path"),
            }
        )
    for event in ranked_events[:3]:
        evidence.append(
            {
                "timestamp_seconds": float(event.get("timestamp_seconds") or 0),
                "title": str(event.get("title") or event.get("kind") or "Timeline event"),
                "detail": str(event.get("detail") or ""),
                "frame_path": str(event.get("frame_path") or ""),
                "clip_path": None,
            }
        )
    for frame in ranked_frames[:2]:
        text = " | ".join(str(block.get("text") or "") for block in frame.get("ocr_blocks", [])[:6])
        evidence.append(
            {
                "timestamp_seconds": float(frame.get("timestamp_seconds") or 0),
                "title": "Matching frame text",
                "detail": text or "Frame matched the question context.",
                "frame_path": str(frame.get("frame_path") or ""),
                "clip_path": None,
            }
        )
    return _dedupe_evidence(evidence)[:5]


def _rank_events(events: list[dict[str, Any]], terms: set[str]) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda event: _score_text(
            " ".join(
                [
                    str(event.get("kind") or ""),
                    str(event.get("title") or ""),
                    str(event.get("detail") or ""),
                ]
            ),
            terms,
        ),
        reverse=True,
    )


def _rank_frames(frames: list[dict[str, Any]], terms: set[str]) -> list[dict[str, Any]]:
    return sorted(
        frames,
        key=lambda frame: _score_text(
            " ".join(str(block.get("text") or "") for block in frame.get("ocr_blocks", [])),
            terms,
        ),
        reverse=True,
    )


def _question_terms(question: str) -> set[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "what",
        "why",
        "when",
        "where",
        "did",
        "does",
        "do",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
    }
    return {
        term
        for term in re.findall(r"[a-z0-9]+", question.lower())
        if len(term) > 2 and term not in stopwords
    }


def _score_text(text: str, terms: set[str]) -> int:
    lower = text.lower()
    return sum(1 for term in terms if term in lower)


def _dedupe_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[float, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in evidence:
        key = (round(float(item["timestamp_seconds"]), 3), str(item["frame_path"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped

