"""Tests for the Livepeer live-runner wrapper (no orchestrator needed)."""

import base64

import pytest

from screen_agent_mvp.config import AnalysisConfig
from screen_agent_mvp.demo import write_demo_video
from screen_agent_mvp.livepeer_runner import create_app


@pytest.fixture(scope="module")
def demo_video_b64(tmp_path_factory) -> str:
    path = write_demo_video(tmp_path_factory.mktemp("vid") / "demo.mp4", seconds=4, fps=2.0)
    return base64.b64encode(path.read_bytes()).decode()


def _config() -> AnalysisConfig:
    return AnalysisConfig(force_fallback_models=True)


async def test_analyze_returns_report_bundle(aiohttp_client, demo_video_b64):
    client = await aiohttp_client(create_app(_config()))
    resp = await client.post("/analyze", json={"video_b64": demo_video_b64})
    assert resp.status == 200
    body = await resp.json()
    assert body["report_markdown"].lstrip().startswith("#")
    assert "report" in body["summary"]
    assert "events" in body["timeline"]


async def test_analyze_applies_preset(aiohttp_client, demo_video_b64):
    client = await aiohttp_client(create_app(_config()))
    resp = await client.post(
        "/analyze", json={"video_b64": demo_video_b64, "preset": "support-session"}
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["summary"]["report"]["preset"] == "support-session"


async def test_analyze_rejects_unknown_preset(aiohttp_client, demo_video_b64):
    client = await aiohttp_client(create_app(_config()))
    resp = await client.post(
        "/analyze", json={"video_b64": demo_video_b64, "preset": "nonsense"}
    )
    assert resp.status == 400


async def test_analyze_rejects_missing_video(aiohttp_client):
    client = await aiohttp_client(create_app(_config()))
    resp = await client.post("/analyze", json={})
    assert resp.status == 400


async def test_analyze_rejects_bad_base64(aiohttp_client):
    client = await aiohttp_client(create_app(_config()))
    resp = await client.post("/analyze", json={"video_b64": "not base64!!!"})
    assert resp.status == 400


async def test_health(aiohttp_client):
    client = await aiohttp_client(create_app(_config()))
    resp = await client.get("/health")
    assert resp.status == 200
    body = await resp.json()
    assert body["app"] == "emran/screen-agent"


async def test_capability_served_when_configured(aiohttp_client):
    descriptor = {"capability": {"semantic_key": "live-runner:video-understanding:screen-agent:-"}}
    client = await aiohttp_client(create_app(_config(), capability=descriptor))
    resp = await client.get("/capability")
    assert resp.status == 200
    assert await resp.json() == descriptor


async def test_capability_404_when_absent(aiohttp_client):
    client = await aiohttp_client(create_app(_config()))
    resp = await client.get("/capability")
    assert resp.status == 404
