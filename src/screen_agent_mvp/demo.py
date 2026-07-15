from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_demo_video(path: str | Path, seconds: int = 8, fps: float = 2.0) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 720
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not create demo video: {target}")
    try:
        total_frames = int(seconds * fps)
        for index in range(total_frames):
            timestamp = index / fps
            frame = _draw_demo_frame(width, height, timestamp)
            writer.write(frame)
    finally:
        writer.release()
    return target


def _draw_demo_frame(width: int, height: int, timestamp: float) -> np.ndarray:
    frame = np.full((height, width, 3), (248, 249, 251), dtype=np.uint8)
    _rect(frame, 0, 0, width, 72, (32, 37, 48), fill=True)
    _text(frame, "Acme Checkout", 32, 46, 1.0, (255, 255, 255), 2)
    _rect(frame, 80, 120, 760, 640, (255, 255, 255), fill=True)
    _rect(frame, 80, 120, 760, 640, (222, 226, 232), fill=False, thickness=2)
    _text(frame, "Payment", 120, 180, 1.1, (20, 24, 32), 2)
    _text(frame, "Card number", 120, 245, 0.7, (80, 86, 96), 2)
    _rect(frame, 120, 265, 700, 320, (245, 247, 250), fill=True)
    _rect(frame, 120, 265, 700, 320, (188, 196, 208), fill=False, thickness=2)
    _text(frame, "4242 4242 4242 4242", 140, 302, 0.8, (30, 35, 45), 2)
    _text(frame, "ZIP code", 120, 375, 0.7, (80, 86, 96), 2)
    _rect(frame, 120, 395, 420, 450, (245, 247, 250), fill=True)
    _rect(frame, 120, 395, 420, 450, (188, 196, 208), fill=False, thickness=2)

    if timestamp < 4:
        _text(frame, "Click Pay to complete order", 120, 525, 0.8, (80, 86, 96), 2)
        button_color = (28, 105, 245)
        button_text = "Pay $49"
    else:
        _text(frame, "Error: ZIP code is required", 120, 485, 0.85, (36, 36, 220), 2)
        button_color = (155, 164, 178)
        button_text = "Pay disabled"

    _rect(frame, 120, 545, 360, 610, button_color, fill=True)
    _text(frame, button_text, 152, 588, 0.85, (255, 255, 255), 2)
    _rect(frame, 840, 120, 1200, 350, (255, 255, 255), fill=True)
    _rect(frame, 840, 120, 1200, 350, (222, 226, 232), fill=False, thickness=2)
    _text(frame, "Order Summary", 880, 180, 0.9, (20, 24, 32), 2)
    _text(frame, "Starter plan", 880, 245, 0.75, (80, 86, 96), 2)
    _text(frame, "$49.00", 880, 300, 0.9, (20, 24, 32), 2)
    return frame


def _rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    *,
    fill: bool,
    thickness: int = 1,
) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1 if fill else thickness)


def _text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

