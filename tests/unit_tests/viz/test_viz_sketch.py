"""Tests for viz architecture sketch."""

from viz.app import create_app


def test_create_app_has_health_endpoint():
    app = create_app()

    assert any(route.path == "/health" for route in app.routes)
