from __future__ import annotations

from screen_agent_mvp.web import build_app


def test_build_app_returns_gradio_blocks() -> None:
    app = build_app()

    assert app.__class__.__name__ == "Blocks"

