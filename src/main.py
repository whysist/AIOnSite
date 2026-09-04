"""Application entry point.

Usage::

    python -m src.main                 # print resolved config + provider health
    python -m src.main --serve         # run the FastAPI app via uvicorn
    python -m src.main --task "..."    # run one task end-to-end and print JSON

Secrets are never printed -- config is shown via ``Settings.safe_dump()``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .core.config import get_settings
from .core.exceptions import ConfigurationError
from .core.logging import configure_logging, get_logger

_log = get_logger("main")


async def _startup_report() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    print("AIOnSite -- Sovereign On-Premise Agentic AI Workbench")
    print(f"  environment    : {settings.environment.value}")
    print(f"  llm_provider   : {settings.llm_provider.value}")
    print(f"  sovereign_mode : {settings.sovereign_mode}")
    print(f"  llm_model      : {settings.llm_model or '(provider default)'}")

    from .llm.factory import create_llm

    llm = create_llm(settings)
    try:
        healthy = await llm.health_check()
    finally:
        await llm.aclose()
    print(f"  provider health: {'reachable' if healthy else 'UNREACHABLE'}")
    return 0 if healthy or settings.llm_provider.value == "echo" else 1


async def _run_task(task: str, *, confidential: bool) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    from .orchestrator import Orchestrator

    orch = Orchestrator(settings)
    try:
        ctx = await orch.run_task(task, confidential=confidential)
    finally:
        await orch.aclose()
    print(json.dumps(ctx.summary(), indent=2, default=str))
    return 0 if ctx.status.value in ("completed", "needs_review") else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.main")
    parser.add_argument("--serve", action="store_true", help="run the HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", help="run a single task end-to-end and exit")
    parser.add_argument("--confidential", action="store_true",
                        help="force local-only routing for --task")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        get_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.serve:
        import uvicorn

        uvicorn.run(
            "src.api.app:app", host=args.host, port=args.port,
            log_level=get_settings().log_level.lower(),
        )
        return 0
    if args.task:
        return asyncio.run(_run_task(args.task, confidential=args.confidential))
    return asyncio.run(_startup_report())


if __name__ == "__main__":
    raise SystemExit(main())
