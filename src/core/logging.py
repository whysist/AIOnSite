import logging

import structlog


def configure_logging(level: str = "INFO"):
    logging.basicConfig(level=level)

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )
