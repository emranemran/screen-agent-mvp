from __future__ import annotations

import json
import re
import shlex
import subprocess
from typing import Protocol

import cv2
import numpy as np

from .config import AnalysisConfig
from .schema import FrameObservation, OcrBlock, Report, UiElement


class OcrEngine(Protocol):
    def detect(self, image_path: str) -> tuple[list[OcrBlock], list[str]]:
        ...


class UiEngine(Protocol):
    def parse(self, image_path: str, ocr_blocks: list[OcrBlock]) -> tuple[list[UiElement], list[str]]:
        ...


class ReasoningEngine(Protocol):
    def build_report(self, observations: list[FrameObservation]) -> Report:
        ...


class PaddleOcrEngine:
    def __init__(self, strict: bool) -> None:
        self.strict = strict
        self._reader = None

    def _load(self):
        if self._reader is not None:
            return self._reader
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            if self.strict:
                raise RuntimeError("PaddleOCR is required in strict model mode.") from exc
            return None
        self._reader = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            device="cpu",
            enable_mkldnn=False,
        )
        return self._reader

    def detect(self, image_path: str) -> tuple[list[OcrBlock], list[str]]:
        reader = self._load()
        if reader is None:
            return FallbackOcrEngine().detect(image_path)
        warnings: list[str] = []
        blocks: list[OcrBlock] = []
        try:
            results = reader.predict(image_path)
        except AttributeError:
            results = reader.ocr(image_path, cls=False)
        except Exception as exc:
            if self.strict:
                raise
            warnings.append(f"PaddleOCR failed; fallback OCR used: {exc}")
            return FallbackOcrEngine().detect(image_path)

        for result in results or []:
            if isinstance(result, dict):
                texts = result.get("rec_texts") or []
                scores = result.get("rec_scores") or []
                boxes = result.get("rec_boxes")
                if boxes is None:
                    boxes = result.get("dt_polys")
                if boxes is None:
                    boxes = []
                for text, score, box in zip(texts, scores, boxes):
                    blocks.append(_ocr_block_from_box(text, box, float(score or 0)))
            elif isinstance(result, list):
                for item in result:
                    if not item or len(item) < 2:
                        continue
                    text = item[1][0] if isinstance(item[1], (list, tuple)) else str(item[1])
                    score = float(item[1][1]) if isinstance(item[1], (list, tuple)) and len(item[1]) > 1 else 0
                    blocks.append(_ocr_block_from_box(text, item[0], score))
        if not blocks:
            warnings.append("OCR found no text.")
        return blocks, warnings


class FallbackOcrEngine:
    """Tiny deterministic fallback used for smoke tests before PaddleOCR is installed."""

    def detect(self, image_path: str) -> tuple[list[OcrBlock], list[str]]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = float(np.mean(edges > 0))
        warning = (
            "PaddleOCR is not available; fallback OCR only estimates text-like regions "
            "and does not read text."
        )
        if density < 0.01:
            return [], [warning]
        height, width = image.shape[:2]
        return [
            OcrBlock(
                text="[text-like-region]",
                x1=0,
                y1=0,
                x2=float(width),
                y2=float(height),
                confidence=0.1,
            )
        ], [warning]


class OmniParserUiEngine:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.fallback = HeuristicUiEngine()

    def parse(self, image_path: str, ocr_blocks: list[OcrBlock]) -> tuple[list[UiElement], list[str]]:
        if not self.config.omniparser_command:
            if self.config.strict_models:
                raise RuntimeError(
                    "SCREEN_AGENT_OMNIPARSER_COMMAND is required in strict model mode."
                )
            elements, warnings = self.fallback.parse(image_path, ocr_blocks)
            warnings.append("OmniParser command is not configured; heuristic UI parser used.")
            return elements, warnings

        cmd = shlex.split(self.config.omniparser_command) + [image_path]
        try:
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)
            elements = [
                UiElement(
                    label=str(item.get("label") or item.get("text") or ""),
                    kind=str(item.get("kind") or item.get("type") or "ui"),
                    x1=float(item["x1"]),
                    y1=float(item["y1"]),
                    x2=float(item["x2"]),
                    y2=float(item["y2"]),
                    confidence=float(item.get("confidence", 0)),
                )
                for item in payload
            ]
            return elements, []
        except Exception as exc:
            if self.config.strict_models:
                raise RuntimeError("OmniParser command failed.") from exc
            elements, warnings = self.fallback.parse(image_path, ocr_blocks)
            warnings.append(f"OmniParser command failed; heuristic UI parser used: {exc}")
            return elements, warnings


class HeuristicUiEngine:
    def parse(self, image_path: str, ocr_blocks: list[OcrBlock]) -> tuple[list[UiElement], list[str]]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elements: list[UiElement] = []
        height, width = image.shape[:2]
        min_area = max(80, int(width * height * 0.0004))
        for contour in contours[:300]:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < min_area or w < 16 or h < 10:
                continue
            if w > width * 0.95 and h > height * 0.95:
                continue
            label = _label_for_box(x, y, x + w, y + h, ocr_blocks)
            kind = "text-field" if w > h * 4 else "button-or-control"
            elements.append(
                UiElement(
                    label=label,
                    kind=kind,
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                    confidence=0.25,
                )
            )
        return elements[:80], ["Heuristic UI parser used; install OmniParser for real UI parsing."]


class QwenVlReasoningEngine:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:
            if self.config.strict_models:
                raise RuntimeError("Qwen2.5-VL dependencies are required in strict model mode.") from exc
            return None, None
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.vlm_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(self.config.vlm_model)
        self._model = model
        self._processor = processor
        return model, processor

    def build_report(self, observations: list[FrameObservation]) -> Report:
        model, processor = self._load()
        if model is None or processor is None:
            return TemplateReasoningEngine(
                warning="Qwen2.5-VL is not available; template reasoning used."
            ).build_report(observations)

        # The Week 1 implementation keeps the VLM prompt compact and evidence-oriented. The
        # fallback template stays available for smoke tests and for machines before weights land.
        selected = observations[: self.config.max_keyframes_for_vlm]
        prompt = _build_reasoning_prompt(selected)
        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "image": obs.frame_path}
                        for obs in selected
                    ],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            from qwen_vl_utils import process_vision_info

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            generated = model.generate(**inputs, max_new_tokens=1200)
            trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated, strict=False)
            ]
            answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            return _report_from_text(answer, observations, warnings=[])
        except Exception as exc:
            if self.config.strict_models:
                raise RuntimeError("Qwen2.5-VL report generation failed.") from exc
            return TemplateReasoningEngine(
                warning=f"Qwen2.5-VL failed; template reasoning used: {exc}"
            ).build_report(observations)


class TemplateReasoningEngine:
    def __init__(self, warning: str | None = None) -> None:
        self.warning = warning

    def build_report(self, observations: list[FrameObservation]) -> Report:
        all_text = [
            block.text
            for observation in observations
            for block in observation.ocr_blocks
            if block.text.strip()
        ]
        lowered = " ".join(all_text).lower()
        error_terms = ["error", "failed", "invalid", "required", "denied", "oops", "try again"]
        evidence = []
        for observation in observations:
            frame_text = " ".join(block.text for block in observation.ocr_blocks).strip()
            if any(term in frame_text.lower() for term in error_terms):
                evidence.append(
                    _evidence_from_observation(
                        observation,
                        "Possible error state",
                        frame_text[:240] or "Text/error-like state detected.",
                    )
                )
        if not evidence and observations:
            evidence = [
                _evidence_from_observation(observations[0], "Start state", "First sampled frame."),
                _evidence_from_observation(observations[-1], "End state", "Last sampled frame."),
            ]
        warnings = [self.warning] if self.warning else []
        warnings.extend(
            warning
            for observation in observations
            for warning in observation.warnings
            if warning not in warnings
        )
        found_failure = any(term in lowered for term in error_terms)
        return Report(
            title=(
                "Screen recording shows a likely app failure"
                if found_failure
                else "No clear failure detected in screen recording"
            ),
            outcome="failure_suspected" if found_failure else "uncertain",
            summary=(
                "The sampled frames include error-like UI text. Review the evidence timestamps."
                if found_failure
                else "The sampled frames did not contain obvious error text. Review the evidence frames."
            ),
            repro_steps=[
                "Open the application state shown at the first evidence timestamp.",
                "Follow the visible interaction sequence in the recording.",
                "Observe the final state shown in the last evidence frame.",
            ],
            expected="The app should complete the user task or show a recoverable next step.",
            actual=(
                "The app appears to show an error or blocked state."
                if found_failure
                else "The app outcome is unclear from deterministic fallback analysis."
            ),
            evidence=evidence[:6],
            uncertainty=[
                "Template reasoning cannot infer hidden app state.",
                "Install the Week 1 model stack and rerun with --strict-models for VLM reasoning.",
            ],
            warnings=warnings,
        )


def build_engines(config: AnalysisConfig) -> tuple[OcrEngine, UiEngine, ReasoningEngine]:
    if config.force_fallback_models:
        return FallbackOcrEngine(), HeuristicUiEngine(), TemplateReasoningEngine(
            warning="Forced fallback model mode is enabled."
        )
    return PaddleOcrEngine(strict=config.strict_models), OmniParserUiEngine(config), QwenVlReasoningEngine(config)


def _ocr_block_from_box(text: str, box, confidence: float) -> OcrBlock:
    arr = np.asarray(box, dtype=float)
    if arr.ndim == 1 and arr.size >= 4:
        x1, y1, x2, y2 = arr[:4]
    else:
        xs = arr[:, 0]
        ys = arr[:, 1]
        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    return OcrBlock(str(text), float(x1), float(y1), float(x2), float(y2), confidence)


def _label_for_box(x1: int, y1: int, x2: int, y2: int, ocr_blocks: list[OcrBlock]) -> str:
    labels = []
    for block in ocr_blocks:
        center_x = (block.x1 + block.x2) / 2
        center_y = (block.y1 + block.y2) / 2
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            labels.append(block.text)
    return " ".join(labels)[:80] if labels else ""


def _build_reasoning_prompt(observations: list[FrameObservation]) -> str:
    context = []
    for observation in observations:
        text = " | ".join(block.text for block in observation.ocr_blocks[:20])
        elements = ", ".join(
            f"{element.kind}:{element.label or 'unlabeled'}"
            for element in observation.ui_elements[:20]
        )
        context.append(
            f"{observation.index} @ {observation.timestamp_seconds:.2f}s\n"
            f"OCR: {text or '(none)'}\nUI: {elements or '(none)'}"
        )
    return (
        "You are analyzing a screen recording for a software bug report. "
        "Use the screenshots and structured context below. Return concise Markdown with: "
        "Title, Outcome, Summary, Repro Steps, Expected, Actual, Evidence, Uncertainty.\n\n"
        + "\n\n".join(context)
    )


def _report_from_text(text: str, observations: list[FrameObservation], warnings: list[str]) -> Report:
    sections = _parse_markdown_sections(text)
    evidence = [
        _evidence_from_observation(obs, f"Frame {obs.index}", "Referenced by VLM report.")
        for obs in observations[:6]
    ]
    return Report(
        title=sections.get("title") or "VLM-generated screen recording analysis",
        outcome="vlm_generated",
        summary=sections.get("summary") or text.strip(),
        repro_steps=_parse_numbered_lines(sections.get("repro steps", "")),
        expected=sections.get("expected") or "See VLM-generated summary.",
        actual=sections.get("actual") or "See VLM-generated summary.",
        evidence=evidence,
        uncertainty=_parse_bullet_lines(sections.get("uncertainty", ""))
        or ["The VLM summary should be checked against evidence screenshots."],
        warnings=warnings,
    )


def _parse_markdown_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "summary"
    sections[current] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = re.match(r"^#{1,3}\s+(.+?)\s*$", line)
        if heading:
            current = heading.group(1).strip().lower()
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(raw_line)
    parsed = {key: "\n".join(value).strip() for key, value in sections.items()}
    title = parsed.get("title", "")
    if title:
        parsed["title"] = title.splitlines()[0].strip("-:* ")
    return parsed


def _parse_numbered_lines(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:\d+\.|-)\s*", "", line).strip()
        if cleaned:
            items.append(cleaned)
    return items


def _parse_bullet_lines(text: str) -> list[str]:
    return _parse_numbered_lines(text)


def _evidence_from_observation(observation: FrameObservation, title: str, detail: str):
    from .schema import EvidenceItem

    return EvidenceItem(
        timestamp_seconds=observation.timestamp_seconds,
        title=title,
        detail=detail,
        frame_path=observation.frame_path,
    )
