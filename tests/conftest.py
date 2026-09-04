"""Shared test fixtures.

Forces the deterministic offline provider and a testing environment so the
suite never needs a network, an API key, or a running model server.
"""

from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDER", "echo")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("SOVEREIGN_MODE", "false")

import pytest

from src.core.config import reload_settings


@pytest.fixture(autouse=True)
def _fresh_settings():
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def settings():
    return reload_settings()


@pytest.fixture
def echo_llm():
    from src.llm.echo_provider import EchoProvider

    return EchoProvider()
