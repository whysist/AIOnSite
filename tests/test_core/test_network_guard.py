"""NetworkGuard: the code-level proof that a run touched only loopback.

The httpx-transport tests below exist because the raw-socket patch alone
was proven insufficient on this project's own platform: a live run against
the real Ollama server showed the socket-only patch reporting zero hosts
contacted despite a real request succeeding, because Windows' default
asyncio event loop (ProactorEventLoop) connects via IOCP overlapped I/O and
never calls socket.socket.connect/connect_ex. These tests exercise the real
asyncio event loop this project actually runs on, not just the classification
logic, so a regression back to socket-only interception would be caught here.
"""

from __future__ import annotations

import socket

import httpx

from src.core.network_guard import NetworkGuard


def test_loopback_connection_attempt_is_not_a_violation():
    # UDP: connect() only associates the local socket with a remote address
    # -- it never sends a packet or blocks on the network -- so this stays
    # instant and offline regardless of what (if anything) is listening.
    guard = NetworkGuard()
    with guard:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("127.0.0.1", 1))
        finally:
            s.close()

    assert "127.0.0.1" in guard.contacted_hosts
    assert guard.violations == []
    assert guard.is_sovereign is True


def test_non_loopback_connection_attempt_is_recorded_as_a_violation():
    guard = NetworkGuard()
    with guard:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("203.0.113.1", 80))  # TEST-NET-3 (RFC 5737) -- never a real host
        finally:
            s.close()

    assert "203.0.113.1" in guard.violations
    assert guard.is_sovereign is False
    summary = guard.summary()
    assert summary["sovereign"] is False
    assert "203.0.113.1" in summary["external_hosts"]


def test_patch_is_restored_after_the_context_exits():
    original_connect = socket.socket.connect
    original_async = httpx.AsyncHTTPTransport.handle_async_request
    with NetworkGuard():
        pass
    assert socket.socket.connect is original_connect
    assert httpx.AsyncHTTPTransport.handle_async_request is original_async


async def test_real_httpx_async_request_over_the_real_event_loop_is_observed():
    """The regression test for the actual bug found: a real httpx.AsyncClient
    request, over whatever event loop this interpreter actually uses (the
    Windows default is ProactorEventLoop -- see module docstring), must be
    seen. Port 1 on loopback is essentially never listening, so the request
    fails fast without ever leaving the machine; the host must still be
    recorded because NetworkGuard records before delegating to the real
    transport, not after a successful response.
    """
    guard = NetworkGuard()
    with guard:
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                await client.get("http://127.0.0.1:1")
            except httpx.HTTPError:
                pass

    assert "127.0.0.1" in guard.contacted_hosts
    assert guard.violations == []


def test_real_httpx_sync_request_is_observed():
    guard = NetworkGuard()
    with guard:
        with httpx.Client(timeout=5.0) as client:
            try:
                client.get("http://127.0.0.1:1")
            except httpx.HTTPError:
                pass

    assert "127.0.0.1" in guard.contacted_hosts
    assert guard.violations == []
