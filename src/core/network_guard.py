"""Live proof that a run made no non-loopback network connections.

Sovereign mode is already enforced twice before this ever runs: once at
config load (``Settings`` rejects ``sovereign_mode=True`` with a cloud
provider) and once at routing time (``ModelRouter`` raises
:class:`~src.core.exceptions.SovereigntyError` rather than silently
downgrading). Both of those check *intent* -- what the code is configured
to do. ``NetworkGuard`` is a third, independent check: it watches real
outbound traffic while a run is in flight and records literally every host
a connection was attempted to, so "no external calls were made" is
something a run can *prove* about itself, not only something the source
code claims.

Two interception layers, and both matter:

* ``httpx`` transport hook (the primary, reliable layer). Every LLM
  provider and the VLM call in this project goes through ``httpx`` and
  nothing else -- a deliberate architecture decision (see
  PROJECT_CONTEXT.md sec 6: "Providers via httpx, not vendor SDKs"). Hooking
  ``httpx.AsyncHTTPTransport.handle_async_request`` /
  ``httpx.HTTPTransport.handle_request`` catches every real outbound HTTP
  call this codebase can make, regardless of the event loop implementation.
* Raw ``socket.socket.connect``/``connect_ex`` patch (best-effort, kept as
  defence in depth for any future direct-socket code).

The httpx hook exists because the socket-level patch alone is *not
sufficient on this project's own deployment platform*: Windows' default
asyncio event loop (``ProactorEventLoop``) performs async socket connects
via IOCP overlapped I/O, which never calls ``socket.socket.connect``/
``connect_ex`` at the Python level -- confirmed by a live run against the
real Ollama server, which the socket-only patch reported as zero hosts
contacted despite a real request having succeeded. Relying on the socket
patch alone would have made this "proof" silently false on Windows.

This is a code-level, automatable proof (usable in tests and surfaced via
an audit event); it complements, but does not replace, a physical demo-day
proof (a network monitor on screen, or the venue network unplugged) -- see
the sovereignty section of the demo-readiness plan.
"""

from __future__ import annotations

import socket
from typing import Any

import httpx

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0"})  # noqa: S104 - read-only comparison set, not a bind address


class NetworkGuard:
    """Context manager recording every outbound connection attempted while active.

    ``violations`` lists every non-loopback host actually contacted --
    empty means the sovereignty claim held for this run.
    """

    def __init__(self) -> None:
        self.contacted_hosts: set[str] = set()
        self.violations: list[str] = []
        self._orig_connect: Any = None
        self._orig_connect_ex: Any = None
        self._orig_async_request: Any = None
        self._orig_sync_request: Any = None

    def __enter__(self) -> NetworkGuard:
        self._orig_connect = socket.socket.connect
        self._orig_connect_ex = socket.socket.connect_ex
        self._orig_async_request = httpx.AsyncHTTPTransport.handle_async_request
        self._orig_sync_request = httpx.HTTPTransport.handle_request

        def _traced_connect(sock_self: socket.socket, address: Any, *a: Any, **kw: Any) -> Any:
            self._record(self._host_of(address))
            return self._orig_connect(sock_self, address, *a, **kw)

        def _traced_connect_ex(sock_self: socket.socket, address: Any, *a: Any, **kw: Any) -> Any:
            self._record(self._host_of(address))
            return self._orig_connect_ex(sock_self, address, *a, **kw)

        async def _traced_async_request(transport_self: Any, request: httpx.Request) -> httpx.Response:
            self._record(request.url.host)
            return await self._orig_async_request(transport_self, request)

        def _traced_sync_request(transport_self: Any, request: httpx.Request) -> httpx.Response:
            self._record(request.url.host)
            return self._orig_sync_request(transport_self, request)

        # Deliberately patching stdlib/third-party classes at runtime to
        # observe every connection attempt -- mypy sees this as replacing a
        # stricter built-in signature with a permissive wrapper, which is
        # exactly the point (it must transparently forward whatever args
        # any caller passes), not a type mismatch to design around.
        socket.socket.connect = _traced_connect  # type: ignore[assignment]
        socket.socket.connect_ex = _traced_connect_ex  # type: ignore[assignment]
        httpx.AsyncHTTPTransport.handle_async_request = _traced_async_request  # type: ignore[assignment]
        httpx.HTTPTransport.handle_request = _traced_sync_request  # type: ignore[assignment]
        return self

    def __exit__(self, *exc_info: object) -> None:
        socket.socket.connect = self._orig_connect  # type: ignore[assignment]
        socket.socket.connect_ex = self._orig_connect_ex  # type: ignore[assignment]
        httpx.AsyncHTTPTransport.handle_async_request = self._orig_async_request  # type: ignore[assignment]
        httpx.HTTPTransport.handle_request = self._orig_sync_request  # type: ignore[assignment]

    @staticmethod
    def _host_of(address: Any) -> str:
        return address[0] if isinstance(address, tuple) and address else str(address)

    def _record(self, host: str) -> None:
        self.contacted_hosts.add(host)
        if host not in _LOOPBACK_HOSTS:
            self.violations.append(host)

    @property
    def is_sovereign(self) -> bool:
        return not self.violations

    def summary(self) -> dict[str, object]:
        return {
            "hosts_contacted": sorted(self.contacted_hosts),
            "external_hosts": sorted(set(self.violations)),
            "sovereign": self.is_sovereign,
        }
