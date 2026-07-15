from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .schema import SampledFrame, VideoMetadata


def probe_video(path: str | Path) -> VideoMetadata:
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        return VideoMetadata(
            path=str(video_path),
            duration_seconds=round(duration, 3),
            fps=round(fps, 3),
            frame_count=frame_count,
            width=width,
            height=height,
        )
    finally:
        capture.release()


def build_base_sample_times(duration_seconds: float, sample_fps: float, max_frames: int) -> list[float]:
    if duration_seconds <= 0:
        raise ValueError("Video duration could not be determined.")
    if sample_fps <= 0:
        raise ValueError("Sample FPS must be greater than 0.")
    if max_frames <= 0:
        raise ValueError("Max frames must be greater than 0.")
    interval = 1.0 / sample_fps
    times: list[float] = []
    current = 0.0
    while current < duration_seconds and len(times) < max_frames:
        times.append(round(current, 3))
        current += interval
    end_time = round(max(0.0, duration_seconds - 0.1), 3)
    if duration_seconds > 0 and len(times) < max_frames and (not times or times[-1] < end_time):
        times.append(end_time)
    return sorted(set(times))[:max_frames]


def _read_frame_bgr(capture: cv2.VideoCapture, timestamp_seconds: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
    ok, frame = capture.read()
    if not ok or frame is None:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps > 0 and frame_count > 0:
            frame_index = min(frame_count - 1, max(0, int(round(timestamp_seconds * fps))))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame at {timestamp_seconds:.3f}s")
    return frame


def _frame_difference_score(previous: np.ndarray, current: np.ndarray) -> float:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(previous_gray, current_gray)))


def find_scene_change_times(
    path: str | Path,
    metadata: VideoMetadata,
    threshold: float,
    max_extra_frames: int,
) -> list[float]:
    if max_extra_frames <= 0 or metadata.duration_seconds <= 1:
        return []
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        interval = 0.5
        previous = _read_frame_bgr(capture, 0.0)
        changes: list[float] = []
        current_time = interval
        while current_time < metadata.duration_seconds and len(changes) < max_extra_frames:
            frame = _read_frame_bgr(capture, current_time)
            score = _frame_difference_score(previous, frame)
            if score >= threshold:
                changes.append(round(current_time, 3))
                previous = frame
            current_time += interval
        return changes
    finally:
        capture.release()


def sample_video_frames(
    path: str | Path,
    output_dir: str | Path,
    sample_fps: float,
    max_frames: int,
    scene_change_threshold: float,
) -> tuple[VideoMetadata, list[SampledFrame]]:
    metadata = probe_video(path)
    base_times = build_base_sample_times(metadata.duration_seconds, sample_fps, max_frames)
    remaining = max(0, max_frames - len(base_times))
    scene_times = find_scene_change_times(path, metadata, scene_change_threshold, remaining)
    time_reasons: dict[float, str] = {timestamp: "interval" for timestamp in base_times}
    for timestamp in scene_times:
        time_reasons.setdefault(timestamp, "scene_change")
    selected_times = sorted(time_reasons)[:max_frames]

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    sampled: list[SampledFrame] = []
    try:
        for index, timestamp in enumerate(selected_times, start=1):
            frame = _read_frame_bgr(capture, timestamp)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            frame_path = target_dir / f"frame-{index:04d}-{timestamp:.3f}s.jpg"
            image.save(frame_path, quality=92)
            sampled.append(
                SampledFrame(
                    index=index,
                    timestamp_seconds=timestamp,
                    path=str(frame_path),
                    reason=time_reasons[timestamp],
                )
            )
    finally:
        capture.release()
    return metadata, sampled


def write_evidence_clip(
    video_path: str | Path,
    timestamp_seconds: float,
    output_path: str | Path,
    metadata: VideoMetadata,
    padding_seconds: float = 2.0,
    max_duration_seconds: float = 6.0,
) -> Path:
    start = max(0.0, float(timestamp_seconds) - padding_seconds)
    end = min(metadata.duration_seconds, start + max_duration_seconds)
    if end <= start:
        end = min(metadata.duration_seconds, start + 0.5)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = float(metadata.fps or capture.get(cv2.CAP_PROP_FPS) or 2.0)
        width = int(metadata.width or capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(metadata.height or capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise ValueError("Could not determine video frame size for clip export.")
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(target),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(0.1, fps),
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(f"Could not create evidence clip: {target}")
        try:
            frame_count = max(1, int(round((end - start) * fps)))
            for index in range(frame_count):
                timestamp = start + index / fps
                try:
                    writer.write(_read_frame_bgr(capture, timestamp))
                except ValueError:
                    break
        finally:
            writer.release()
        return target
    finally:
        capture.release()
