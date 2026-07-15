from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class SampledFrame:
    index: int
    timestamp_seconds: float
    path: str
    reason: str


@dataclass(frozen=True)
class OcrBlock:
    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(frozen=True)
class UiElement:
    label: str
    kind: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(frozen=True)
class FrameObservation:
    index: int
    timestamp_seconds: float
    frame_path: str
    reason: str
    ocr_blocks: list[OcrBlock]
    ui_elements: list[UiElement]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TimelineEvent:
    timestamp_seconds: float
    kind: str
    title: str
    detail: str
    frame_path: str
    confidence: float


@dataclass(frozen=True)
class EvidenceItem:
    timestamp_seconds: float
    title: str
    detail: str
    frame_path: str
    clip_path: str | None = None


@dataclass(frozen=True)
class Report:
    title: str
    outcome: str
    summary: str
    repro_steps: list[str]
    expected: str
    actual: str
    evidence: list[EvidenceItem]
    uncertainty: list[str]
    warnings: list[str]
    preset: str = "bug-report"


def to_plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value
