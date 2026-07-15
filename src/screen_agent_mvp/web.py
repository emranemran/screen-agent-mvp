from __future__ import annotations

import os
from typing import Any

import gradio as gr

from .analyzer import analyze_video
from .config import PRESETS, AnalysisConfig
from .qa import answer_run_question, render_answer_markdown
from .reporting import render_markdown_report


def _video_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("name")
    path = getattr(value, "path", None)
    if path:
        return str(path)
    name = getattr(value, "name", None)
    return str(name) if name else None


def run_analysis(video: Any, preset: str, sample_fps: float, max_frames: int, strict_models: bool):
    path = _video_path(video)
    if not path:
        raise gr.Error("Upload a screen recording first.")
    config = AnalysisConfig.from_env(strict_models=strict_models).with_preset(preset)
    config = AnalysisConfig(
        preset=config.preset,
        sample_fps=sample_fps,
        max_frames=max_frames,
        scene_change_threshold=config.scene_change_threshold,
        max_keyframes_for_vlm=config.max_keyframes_for_vlm,
        strict_models=config.strict_models,
        force_fallback_models=config.force_fallback_models,
        vlm_model=config.vlm_model,
        omniparser_command=config.omniparser_command,
    )
    result = analyze_video(path, None, config)
    metadata = result["metadata"]
    observations = result["observations"]
    report = result["report"]
    events = result["events"]
    markdown = render_markdown_report(report, metadata, observations)
    gallery = [
        (observation.frame_path, f"{observation.timestamp_seconds:.2f}s")
        for observation in observations[:24]
    ]
    timeline_rows = [
        [
            f"{event.timestamp_seconds:.2f}",
            event.kind,
            event.title,
            event.detail,
            event.frame_path,
        ]
        for event in events
    ]
    return (
        markdown,
        timeline_rows,
        gallery,
        result["artifacts"],
        result["artifacts"]["report"],
        result["run_dir"],
    )


def ask_question(run_dir: str, question: str) -> str:
    if not run_dir:
        raise gr.Error("Analyze a recording first.")
    try:
        return render_answer_markdown(answer_run_question(run_dir, question))
    except Exception as exc:
        raise gr.Error(f"Question answering failed: {exc}") from exc


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Screen Agent MVP") as app:
        gr.Markdown(
            """
# Screen Agent MVP
Upload a screen recording and generate a local bug-report-style analysis with timestamped evidence.
"""
        )
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Screen recording")
                preset = gr.Dropdown(PRESETS, value="bug-report", label="Preset")
                sample_fps = gr.Slider(0.2, 3.0, value=1.0, step=0.2, label="Sample FPS")
                max_frames = gr.Slider(5, 120, value=60, step=1, label="Max frames")
                strict_models = gr.Checkbox(
                    value=False,
                    label="Strict model mode",
                    info="Require PaddleOCR, OmniParser command, and Qwen2.5-VL.",
                )
                button = gr.Button("Analyze recording", variant="primary")
            with gr.Column():
                report = gr.Markdown(label="Report")
                timeline = gr.Dataframe(
                    headers=["Time", "Kind", "Title", "Detail", "Frame"],
                    datatype=["str", "str", "str", "str", "str"],
                    label="Timeline events",
                    interactive=False,
                )
                gallery = gr.Gallery(label="Evidence frames", columns=2, height=400)
                artifacts = gr.JSON(label="Artifacts")
                report_path = gr.Textbox(label="Report path")
                run_dir = gr.Textbox(label="Run directory", visible=False)
                question = gr.Textbox(
                    label="Ask about this run",
                    placeholder="Why did checkout fail? What timestamp shows the error?",
                )
                ask_button = gr.Button("Ask", variant="secondary")
                answer = gr.Markdown(label="Answer")
        button.click(
            run_analysis,
            inputs=[video, preset, sample_fps, max_frames, strict_models],
            outputs=[report, timeline, gallery, artifacts, report_path, run_dir],
        )
        ask_button.click(ask_question, inputs=[run_dir, question], outputs=[answer])
    return app


def main() -> None:
    host = os.getenv("SCREEN_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("SCREEN_AGENT_PORT", "7860"))
    build_app().queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
