"""Tests for viz architecture sketch."""

import pytest

from viz.app import create_app


def test_create_app_not_implemented():
    with pytest.raises(NotImplementedError):
        create_app()
