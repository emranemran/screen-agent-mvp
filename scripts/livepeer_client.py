#!/usr/bin/env python3
"""Call screen-agent through the Livepeer orchestrator: reserve → analyze → settle.

Livepeer integration (grep `# Livepeer:`):
  1. reserve_session()      — discover + reserve an orchestrator advertising the app
  2. call_runner()          — POST the video through the orchestrator's proxy
  3. stop_runner_session()  — end the session

Usage:
  uv run python scripts/livepeer_client.py recording.mp4 --out livepeer-run
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
from contextlib import suppress
from pathlib import Path

from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.live_runner import call_runner, stop_runner_session
from livepeer_gateway.selection import reserve_session

APP_ID = "emran/screen-agent"
# Analysis of a real recording takes tens of seconds (minutes in strict mode);
# the SDK default timeout is 5s, which would kill every call mid-analysis.
CALL_TIMEOUT_S = 600.0

log = logging.getLogger("screen-agent-client")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a screen recording via Livepeer.")
    parser.add_argument("video", type=Path, help="Local .mp4/.webm screen recording")
    parser.add_argument("--discovery", default="https://localhost:8935/discovery")
    parser.add_argument("--app", default=APP_ID, help="App id to reserve (matches --app-id on the runner).")
    parser.add_argument("--preset", default="bug-report")
    parser.add_argument("--out", type=Path, default=Path("livepeer-run"))
    parser.add_argument("--signer", default="", help="Remote signer base URL (on-chain path).")
    parser.add_argument(
        "--signer-key",
        default=os.getenv("DAYDREAM_API_KEY", ""),
        help="Bearer token for the remote signer (defaults to $DAYDREAM_API_KEY). "
        "Required by signers with an auth front door, e.g. signer.daydream.live.",
    )
    return parser.parse_args()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    signer_url = args.signer.strip() or None
    signer_headers = (
        {"Authorization": f"Bearer {args.signer_key.strip()}"}
        if signer_url and args.signer_key.strip()
        else None
    )
    session = None
    try:
        session = await reserve_session(  # Livepeer: 1
            discovery_url=args.discovery,
            app=args.app,
            signer_url=signer_url,
            signer_headers=signer_headers,
        )
        log.info("session_id=%s app_url=%s", session.session_id, session.app_url)

        result = await call_runner(  # Livepeer: 2
            runner_url=session.app_url.rstrip("/") + "/analyze",
            payload={
                "video_b64": base64.b64encode(args.video.read_bytes()).decode(),
                "preset": args.preset,
            },
            signer_url=signer_url,
            signer_headers=signer_headers,
            timeout=CALL_TIMEOUT_S,
        )
        data = result.data
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.md").write_text(data["report_markdown"], encoding="utf-8")
        (args.out / "summary.json").write_text(json.dumps(data["summary"], indent=2))
        (args.out / "timeline.json").write_text(json.dumps(data["timeline"], indent=2))
        print(data["report_markdown"])
        print(f"\nSaved bundle to {args.out}/")
    except LivepeerGatewayError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    finally:
        if session is not None:
            with suppress(Exception):
                await stop_runner_session(session)  # Livepeer: 3


if __name__ == "__main__":
    asyncio.run(main())
