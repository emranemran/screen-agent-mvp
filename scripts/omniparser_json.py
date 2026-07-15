#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import types
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit OmniParser UI elements as JSON.")
    parser.add_argument("image", help="Screenshot image path")
    parser.add_argument(
        "--omniparser-root",
        default=os.getenv("OMNIPARSER_ROOT", "/home/codexssh/OmniParser"),
        help="Path to the OmniParser checkout",
    )
    parser.add_argument("--box-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--use-paddleocr",
        action="store_true",
        help="Use OmniParser's PaddleOCR path instead of EasyOCR.",
    )
    parser.add_argument(
        "--caption-icons",
        action="store_true",
        help="Caption icons with Florence. Slower and more version-sensitive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.omniparser_root)
    if not root.exists():
        raise SystemExit(f"OmniParser root not found: {root}")
    sys.path.insert(0, str(root))
    if not args.use_paddleocr:
        _install_paddleocr_import_shim()

    # OmniParser's utilities print progress to stdout; keep stdout JSON-only for callers.
    with contextlib.redirect_stdout(sys.stderr):
        from util.utils import (
            check_ocr_box,
            get_som_labeled_img,
            get_yolo_model,
        )

        image = Image.open(args.image).convert("RGB")
        width, height = image.size
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            "text_scale": 0.8 * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), 1),
            "text_padding": max(int(3 * box_overlay_ratio), 1),
            "thickness": max(int(3 * box_overlay_ratio), 1),
        }
        yolo_model = get_yolo_model(str(root / "weights/icon_detect/model.pt"))
        caption = (
            _load_florence_caption(str(root / "weights/icon_caption_florence"))
            if args.caption_icons
            else None
        )
        ocr_result, _ = check_ocr_box(
            image,
            display_img=False,
            output_bb_format="xyxy",
            goal_filtering=None,
            easyocr_args={"paragraph": False, "text_threshold": 0.9},
            use_paddleocr=args.use_paddleocr,
        )
        text, ocr_bbox = ocr_result
        _, _, parsed = get_som_labeled_img(
            image,
            yolo_model,
            BOX_TRESHOLD=args.box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption,
            ocr_text=text,
            use_local_semantics=args.caption_icons,
            iou_threshold=args.iou_threshold,
            imgsz=args.imgsz,
        )

    elements = []
    for item in parsed:
        bbox = item.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        elements.append(
            {
                "label": str(item.get("content") or ""),
                "kind": str(item.get("type") or "ui"),
                "x1": float(x1) * width,
                "y1": float(y1) * height,
                "x2": float(x2) * width,
                "y2": float(y2) * height,
                "confidence": 0.8 if item.get("source") else 0.6,
            }
        )
    json.dump(elements, sys.stdout)


def _load_florence_caption(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True,
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    return {"model": model, "processor": processor}


def _install_paddleocr_import_shim() -> None:
    """Avoid OmniParser's import-time PaddleOCR 2.x construction on PaddleOCR 3.x.

    OmniParser's `util.utils` creates a global PaddleOCR reader during import with
    arguments that are no longer accepted by PaddleOCR 3.x. The wrapper defaults to
    EasyOCR for OmniParser text boxes, so the global PaddleOCR object is not used.
    """
    fake = types.ModuleType("paddleocr")

    class _UnusedPaddleOCR:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def ocr(self, *args, **kwargs):
            raise RuntimeError("PaddleOCR shim is only valid when --use-paddleocr is false.")

    fake.PaddleOCR = _UnusedPaddleOCR
    sys.modules["paddleocr"] = fake


if __name__ == "__main__":
    main()
