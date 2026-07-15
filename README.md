# Screen Agent MVP

Fresh local MVP for turning screen recordings into timestamped bug-report evidence.

This project is intentionally separate from `locate-anything-local`. Week 1 focuses on a local,
single-machine demo for an RTX 5090 running Ubuntu/WSL2:

1. Ingest a local `.mp4` or `.webm`.
2. Sample frames at 1 FPS plus scene changes.
3. Run OCR and UI parsing.
4. Ask a local VLM to produce a bug-report-style summary.
5. Write `report.md`, `timeline.json`, screenshots, and summary JSON.

The code has deterministic fallback engines so the pipeline and UI can be tested before model
weights are installed. For the real 5090 path, install the `models` dependency group and run with
`--strict-models`.

## WSL2 Setup

Use Ubuntu/WSL2 with the 5090 visible through `nvidia-smi`.

```bash
cd "/mnt/c/Users/emran/OneDrive/Documents/New project/screen-agent-mvp"
uv python install 3.11
uv sync --group dev --group models
uv run screen-agent diagnostics
uv run screen-agent models-check
```

If PyTorch does not pick a CUDA wheel automatically, install the CUDA 12.8+ wheel that matches
your WSL driver, then rerun diagnostics.

## CLI Demo

Create a small synthetic checkout-failure recording:

```bash
uv run screen-agent make-demo-video --out runs/demo-checkout-failure.mp4
```

Analyze it:

```bash
uv run screen-agent analyze path/to/recording.mp4 --out runs/demo-001
```

Use strict model mode when PaddleOCR, OmniParser, and Qwen2.5-VL are installed:

```bash
uv run screen-agent analyze path/to/recording.mp4 --out runs/demo-001 --strict-models
```

If setup is incomplete, run:

```bash
uv run screen-agent models-check --strict
```

It exits nonzero and reports which required local component is missing.

Outputs:

- `report.md`
- `timeline.json`
- `summary.json`
- `keyframes/*.jpg`
- `evidence_clips/*.mp4`

Architecture notes:

- Open `docs/architecture.html` for the living block diagrams, pipeline map, model stack,
  data contracts, and phase-by-phase roadmap.

Choose a preset when you want different sampling defaults:

```bash
uv run screen-agent analyze path/to/recording.mp4 --preset agent-eval --out runs/agent-eval-001
```

Available presets:

- `bug-report`: default for short failed-user-session demos.
- `support-session`: lower sample rate and more total frames for longer support recordings.
- `agent-eval`: denser sampling for browser-agent task verification.

Ask questions over an existing run without rerunning models:

```bash
uv run screen-agent ask runs/demo-001 "what timestamp shows the error?"
uv run screen-agent ask runs/demo-001 "why did checkout fail?"
```

## Web Demo

```bash
uv run screen-agent-web
```

Open <http://localhost:7860>.

After an analysis completes, use the **Ask about this run** box to query the stored report and
timeline artifacts.

## Running from Your Existing Ubuntu/WSL2 Shell

If Codex/PowerShell cannot see your WSL distro but your Ubuntu shell works, run this inside that
Ubuntu terminal:

```bash
cd "/mnt/c/Users/emran/OneDrive/Documents/New project/screen-agent-mvp"
uv python install 3.11
uv sync --group dev --group models
uv run screen-agent diagnostics
uv run screen-agent models-check
uv run screen-agent make-demo-video --out runs/demo-checkout-failure.mp4
uv run screen-agent analyze runs/demo-checkout-failure.mp4 --out runs/demo-fallback
```

After configuring OmniParser and confirming CUDA/PaddleOCR/Qwen are ready:

```bash
uv run screen-agent models-check --strict
uv run screen-agent analyze runs/demo-checkout-failure.mp4 --out runs/demo-strict --strict-models
```

## Model Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SCREEN_AGENT_VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Hugging Face VLM ID or local path |
| `SCREEN_AGENT_MAX_FRAMES` | `120` | Default maximum sampled frames |
| `SCREEN_AGENT_SAMPLE_FPS` | `1.0` | Base sampling rate |
| `SCREEN_AGENT_OMNIPARSER_COMMAND` | empty | Optional command for a local OmniParser wrapper |

OmniParser is not distributed as a normal Python package in this MVP. To use it in strict mode,
provide a command that accepts an image path and writes JSON UI elements to stdout:

```json
[
  {"label": "Pay", "kind": "button", "x1": 10, "y1": 20, "x2": 120, "y2": 60, "confidence": 0.91}
]
```

Without that command, the MVP uses a local contour/OCR-based UI parser and records a warning.
