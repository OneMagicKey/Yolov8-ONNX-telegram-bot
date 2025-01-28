import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message


class LoggingMiddleware(BaseMiddleware):
    """
    Middleware to log which function handled an event
    """

    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        result = await handler(event, data)
        duration = loop.time() - start_time

        logging.info(
            msg=f"Handler {data['handler'].callback} took {int(duration * 1000)} ms"
        )

        return result


logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("aiogram.event").setLevel(logging.WARNING)  # disable redundant aiogram.events # fmt: skip
