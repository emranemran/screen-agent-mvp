# MVP Roadmap

## Week 1: Basic Local Demo

Build the first local proof that a screen recording can become bug-report evidence.

Deliverables:

- CLI analysis command.
- Gradio upload-and-report demo.
- Video metadata probing and frame sampling.
- Keyframe extraction at 1 FPS plus scene changes.
- OCR blocks per frame.
- UI elements per frame.
- VLM-generated bug report.
- Markdown and JSON artifacts.

Acceptance criteria:

- `screen-agent analyze input.mp4 --out runs/demo --strict-models` completes on the 5090.
- `screen-agent-web` launches and can analyze the same video from the browser.
- Output includes `report.md`, `timeline.json`, `summary.json`, and `keyframes/`.
- Report contains timestamped evidence and uncertainty notes.

## Week 2: Better Demo UX

Make the demo feel like a useful app instead of only a pipeline.

Deliverables:

- Timeline UI with clickable keyframes.
- Evidence clip export around important timestamps.
- Presets for `bug-report`, `support-session`, and `agent-eval`.
- Stable structured event schema for downstream APIs.

Current implementation status:

- Timeline events are derived from new OCR text, possible error states, scene changes, and final state.
- Evidence clips are exported beside report evidence.
- The web UI shows report, timeline rows, keyframes, artifact paths, and report path.
- The CLI and web UI can answer questions over a completed run's stored artifacts.

Acceptance criteria:

- User can identify the suspected failure without watching the whole video.
- Evidence frames and clips open from the report UI.
- `timeline.json` is stable enough for API consumers.

## Week 3: Browser Recorder and Agent Evaluation

Add the strongest agentic angle: verify what browser agents actually did from pixels.

Deliverables:

- Local browser recorder or Playwright recorder.
- Optional URL, click-coordinate, viewport, DOM title, and console-error telemetry.
- Agent task verdict: success, failure, or uncertain.
- Before/after comparison mode.

Acceptance criteria:

- A browser session can be recorded locally and analyzed.
- The report can answer whether a task was completed.
- Two recordings can be compared for behavior differences.

## Week 4: Local Developer API

Expose the MVP as a local API.

Deliverables:

- FastAPI app.
- `POST /runs`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/timeline`
- `POST /runs/{run_id}/ask`
- `GET /runs/{run_id}/report.md`
- `GET /runs/{run_id}/evidence/{id}`

Acceptance criteria:

- A script can upload a video, poll status, and fetch the report.
- Multiple runs do not overwrite one another.
- Bad videos, OOMs, and missing model setup return clear errors.

## Week 5: Evaluation Gate

Decide whether the idea is worth more time or money.

Test scenarios:

- Failed checkout.
- Failed signup/login.
- Broken onboarding.
- Support screen-share recording.
- AI browser-agent task attempt.
- Successful normal task.
- Noisy or confusing session.

Metrics:

- Report usefulness.
- Timestamp accuracy.
- OCR quality.
- UI parser quality.
- VLM hallucination rate.
- Runtime per minute of video.
- VRAM use.
