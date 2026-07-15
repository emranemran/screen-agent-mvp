from __future__ import annotations

import json
import platform
import shlex
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .analyzer import analyze_video, summarize_result
from .config import PRESETS, AnalysisConfig
from .demo import write_demo_video
from .qa import answer_run_question, render_answer_markdown
from .schema import to_plain

app = typer.Typer(help="Local screen recording agent MVP.")
console = Console()


@app.command()
def diagnostics() -> None:
    """Print local runtime diagnostics."""
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": _torch_cuda_info(),
    }
    console.print_json(json.dumps(payload))


@app.command("models-check")
def models_check(
    strict: Annotated[bool, typer.Option(help="Exit nonzero if any required model check fails")] = False,
) -> None:
    """Check readiness for the strict local model path."""
    checks = {
        "torch_cuda": _check_torch_cuda(),
        "paddleocr": _check_import("paddleocr", "PaddleOCR package"),
        "qwen_transformers": _check_qwen_transformers(),
        "qwen_vl_utils": _check_import("qwen_vl_utils", "qwen-vl-utils package"),
        "omniparser_command": _check_omniparser_command(),
    }
    console.print_json(json.dumps(checks))
    if strict and not all(item["ok"] for item in checks.values()):
        raise typer.Exit(code=1)


@app.command()
def analyze(
    video: Annotated[Path, typer.Argument(help="Local .mp4/.webm screen recording")],
    out: Annotated[Path | None, typer.Option("--out", help="Output run directory")] = None,
    sample_fps: Annotated[float, typer.Option(help="Base frame sample rate")] = 1.0,
    max_frames: Annotated[int, typer.Option(help="Maximum sampled frames")] = 120,
    preset: Annotated[
        str,
        typer.Option(help=f"Analysis preset: {', '.join(PRESETS)}"),
    ] = "bug-report",
    strict_models: Annotated[
        bool,
        typer.Option(help="Require PaddleOCR, OmniParser command, and Qwen2.5-VL"),
    ] = False,
    force_fallback: Annotated[
        bool,
        typer.Option(help="Use deterministic fallback engines even when model deps are installed"),
    ] = False,
) -> None:
    """Analyze a screen recording and write report artifacts."""
    config = AnalysisConfig.from_env(strict_models=strict_models).with_preset(preset)
    config = AnalysisConfig(
        preset=config.preset,
        sample_fps=sample_fps,
        max_frames=max_frames,
        scene_change_threshold=config.scene_change_threshold,
        max_keyframes_for_vlm=config.max_keyframes_for_vlm,
        strict_models=config.strict_models,
        force_fallback_models=force_fallback,
        vlm_model=config.vlm_model,
        omniparser_command=config.omniparser_command,
    )
    result = analyze_video(video, out, config)
    console.print(summarize_result(result))
    console.print_json(json.dumps(to_plain(result["artifacts"])))


@app.command("make-demo-video")
def make_demo_video(
    out: Annotated[
        Path,
        typer.Option("--out", help="Output demo video path"),
    ] = Path("runs/demo-checkout-failure.mp4"),
) -> None:
    """Create a small synthetic checkout-failure screen recording."""
    path = write_demo_video(out)
    console.print(str(path))


@app.command()
def ask(
    run_dir: Annotated[Path, typer.Argument(help="Analysis run directory")],
    question: Annotated[str, typer.Argument(help="Question to answer from report/timeline artifacts")],
) -> None:
    """Answer a question using a completed run's local artifacts."""
    result = answer_run_question(run_dir, question)
    console.print(render_answer_markdown(result))
    console.print_json(json.dumps(to_plain(result)))


def _torch_cuda_info() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:
        return {"torch_available": False, "error": str(exc)}
    info: dict[str, object] = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "device": torch.cuda.get_device_name(device),
                "total_vram_gib": round(props.total_memory / 1024**3, 2),
            }
        )
    return info


def _check_import(module: str, label: str) -> dict[str, object]:
    try:
        __import__(module)
        return {"ok": True, "label": label}
    except Exception as exc:
        return {"ok": False, "label": label, "error": str(exc)}


def _check_torch_cuda() -> dict[str, object]:
    info = _torch_cuda_info()
    return {
        "ok": bool(info.get("torch_available")) and bool(info.get("cuda_available")),
        "label": "PyTorch CUDA",
        "details": info,
    }


def _check_qwen_transformers() -> dict[str, object]:
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        return {
            "ok": True,
            "label": "Qwen2.5-VL transformers classes",
            "classes": [
                AutoProcessor.__name__,
                Qwen2_5_VLForConditionalGeneration.__name__,
            ],
        }
    except Exception as exc:
        return {"ok": False, "label": "Qwen2.5-VL transformers classes", "error": str(exc)}


def _check_omniparser_command() -> dict[str, object]:
    import os

    command = os.getenv("SCREEN_AGENT_OMNIPARSER_COMMAND")
    if not command:
        return {
            "ok": False,
            "label": "OmniParser command",
            "error": "SCREEN_AGENT_OMNIPARSER_COMMAND is not set.",
        }
    parts = shlex.split(command)
    if not parts:
        return {
            "ok": False,
            "label": "OmniParser command",
            "error": "SCREEN_AGENT_OMNIPARSER_COMMAND is empty.",
        }
    executable = parts[0]
    resolved = shutil.which(executable)
    if resolved is None and not Path(executable).exists():
        return {
            "ok": False,
            "label": "OmniParser command",
            "command": command,
            "error": f"Executable not found: {executable}",
        }
    return {"ok": True, "label": "OmniParser command", "command": command}


if __name__ == "__main__":
    app()
