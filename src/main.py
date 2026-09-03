import asyncio

from src.core.config import settings
from src.core.logging import configure_logging


async def main():
    configure_logging(settings.log_level)

    print("AIOnSite initialized")
    print(f"Environment: {settings.environment}")
    print(f"LLM Provider: {settings.llm_provider}")


if __name__ == "__main__":
    asyncio.run(main())
