"""Livepeer live-runner wrapper: analyze_video() as a network-callable app.

Livepeer integration (grep `# Livepeer:`):
  1. register_runner()     — announce the app to the orchestrator (startup)
  2. registration.close()  — deregister (cleanup)

/analyze is an ordinary HTTP handler; the pipeline itself is untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import logging
import tempfile
from contextlib import suppress
from pathlib import Path

from aiohttp import web

from .analyzer import analyze_video
from .config import AnalysisConfig

APP_ID = "emran/screen-agent"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8989
# base64 inflates ~4/3; 256 MiB accommodates a ~190 MiB screen recording.
MAX_REQUEST_BYTES = 256 * 1024 * 1024

log = logging.getLogger("screen-agent-runner")


async def _handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "app": request.app.get("app_id", APP_ID)})


async def _handle_capability(request: web.Request) -> web.Response:
    capability = request.app.get("capability")
    if capability is None:
        raise web.HTTPNotFound(text="no capability descriptor configured")
    return web.json_response(capability)


async def _handle_analyze(request: web.Request) -> web.Response:
    payload = await request.json()
    video_b64 = payload.get("video_b64")
    if not isinstance(video_b64, str) or not video_b64:
        raise web.HTTPBadRequest(text='"video_b64" (base64-encoded mp4/webm) is required')
    try:
        video_bytes = base64.b64decode(video_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise web.HTTPBadRequest(text=f"invalid base64 video: {exc}") from exc

    config: AnalysisConfig = request.app["config"]
    preset = payload.get("preset")
    if preset:
        try:
            config = config.with_preset(str(preset))
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix="screen-agent-") as tmp:
        video_path = Path(tmp) / "input.mp4"
        video_path.write_bytes(video_bytes)
        # analyze_video is synchronous/CPU-bound; keep the event loop
        # (registration heartbeats, health checks) alive while it grinds.
        result = await asyncio.to_thread(analyze_video, video_path, Path(tmp) / "run", config)
        artifacts = result["artifacts"]
        response = {
            "report_markdown": Path(artifacts["report"]).read_text(encoding="utf-8"),
            "summary": json.loads(Path(artifacts["summary"]).read_text(encoding="utf-8")),
            "timeline": json.loads(Path(artifacts["timeline"]).read_text(encoding="utf-8")),
        }
    return web.json_response(response)


def create_app(config: AnalysisConfig, capability: dict | None = None) -> web.Application:
    """Build the aiohttp app without registration — used by tests directly."""
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    app["config"] = config
    if capability is not None:
        app["capability"] = capability
    app.router.add_post("/analyze", _handle_analyze)
    app.router.add_get("/health", _handle_health)
    app.router.add_get("/capability", _handle_capability)
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="screen-agent as a Livepeer live runner.")
    parser.add_argument("--orchestrator", default="https://localhost:8935")
    parser.add_argument("--orchSecret", default="abcdef")
    parser.add_argument("--runner-url", default=f"http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Bind address (use 0.0.0.0 when the orchestrator is in Docker).",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--app-id",
        default=APP_ID,
        help="App id advertised in discovery (e.g. livepeer-example/screen-agent).",
    )
    parser.add_argument(
        "--price",
        type=int,
        default=0,
        help="Price in USD per pixels-per-unit (0 = free, the offchain default).",
    )
    parser.add_argument(
        "--pixels-per-unit",
        type=int,
        default=1,
        help="Scale factor: price is charged per this many units.",
    )
    parser.add_argument(
        "--strict-models",
        action="store_true",
        help="Require PaddleOCR, OmniParser, and Qwen2.5-VL (no fallbacks).",
    )
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Skip model loading entirely; heuristic/template engines only.",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Serve only; skip dynamic registration (static -liveRunnerConfig posture — "
        "the orchestrator health-polls this runner instead).",
    )
    parser.add_argument(
        "--capability-file",
        default="",
        help="Path to a capability descriptor JSON; served verbatim at GET /capability.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    config = AnalysisConfig.from_env(strict_models=args.strict_models)
    if args.force_fallback:
        config = AnalysisConfig(force_fallback_models=True)

    capability = None
    if args.capability_file:
        capability = json.loads(Path(args.capability_file).read_text(encoding="utf-8"))

    if args.no_register:
        # Static posture: the orchestrator's -liveRunnerConfig health-polls us.
        app = create_app(config, capability=capability)
        app["app_id"] = args.app_id
        web.run_app(app, host=args.host, port=args.port)
        return

    # Import here so create_app() stays usable without the SDK installed.
    from livepeer_gateway.live_runner import register_runner

    async def _on_startup(app: web.Application) -> None:
        app["registration"] = await register_runner(  # Livepeer: 1
            args.orchestrator,
            secret=args.orchSecret,
            runner_url=args.runner_url,
            app=args.app_id,
            # Analysis is single-shot by nature; persistent until single-shot
            # payment lands upstream (go-livepeer#3955). Offchain → free.
            mode="persistent",
            # One analysis at a time — the pipeline saturates CPU/GPU.
            capacity=1,
            price_per_unit=args.price,
            pixels_per_unit=args.pixels_per_unit,
        )
        log.info(
            "registered runner_id=%s orchestrator=%s",
            app["registration"].runner_id,
            args.orchestrator,
        )

    async def _on_cleanup(app: web.Application) -> None:
        with suppress(Exception):
            await app["registration"].close()  # Livepeer: 2

    app = create_app(config, capability=capability)
    app["app_id"] = args.app_id
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
