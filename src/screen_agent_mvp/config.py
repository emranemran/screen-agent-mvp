from __future__ import annotations

import os
from dataclasses import dataclass, replace


PRESETS = ("bug-report", "support-session", "agent-eval")


@dataclass(frozen=True)
class AnalysisConfig:
    preset: str = "bug-report"
    sample_fps: float = 1.0
    max_frames: int = 120
    scene_change_threshold: float = 18.0
    max_keyframes_for_vlm: int = 12
    strict_models: bool = False
    force_fallback_models: bool = False
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    omniparser_command: str | None = None

    def with_preset(self, preset: str) -> "AnalysisConfig":
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Expected one of: {', '.join(PRESETS)}")
        defaults = {
            "bug-report": {"sample_fps": 1.0, "max_frames": 120, "max_keyframes_for_vlm": 12},
            "support-session": {"sample_fps": 0.5, "max_frames": 180, "max_keyframes_for_vlm": 16},
            "agent-eval": {"sample_fps": 1.5, "max_frames": 160, "max_keyframes_for_vlm": 14},
        }[preset]
        return replace(self, preset=preset, **defaults)

    @classmethod
    def from_env(cls, strict_models: bool = False) -> "AnalysisConfig":
        preset = os.getenv("SCREEN_AGENT_PRESET", "bug-report")
        return cls(
            preset=preset,
            sample_fps=float(os.getenv("SCREEN_AGENT_SAMPLE_FPS", "1.0")),
            max_frames=int(os.getenv("SCREEN_AGENT_MAX_FRAMES", "120")),
            strict_models=strict_models,
            force_fallback_models=os.getenv("SCREEN_AGENT_FORCE_FALLBACK", "").lower()
            in {"1", "true", "yes"},
            vlm_model=os.getenv("SCREEN_AGENT_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
            omniparser_command=os.getenv("SCREEN_AGENT_OMNIPARSER_COMMAND") or None,
        ).with_preset(preset)
