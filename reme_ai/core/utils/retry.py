import asyncio
import functools
from typing import Callable

from loguru import logger


def mcp_retry(max_retries: int = 3, timeout: float | None = None):

    def decorator(func: Callable):

        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            retries = max_retries if max_retries is not None else getattr(self, "max_retries", 3)
            wait_time = timeout if timeout is not None else getattr(self, "timeout", None)

            # 动态生成操作名称逻辑
            op_name = func.__name__
            if op_name == "call_tool" and args:
                op_name = f"call_tool:{args[0]}"

            last_exc = None
            for i in range(retries):
                try:
                    coro = func(self, *args, **kwargs)
                    if wait_time is not None:
                        return await asyncio.wait_for(coro, timeout=wait_time)
                    return await coro

                except (asyncio.TimeoutError, Exception) as e:
                    last_exc = e
                    logger.warning(
                        f"[{getattr(self, 'name', 'Unknown')}] {op_name} "
                        f"attempt {i + 1}/{retries} failed: {repr(e)}",
                    )

                    if i < retries - 1:
                        # 指数退避 (1, 2, 3...)
                        await asyncio.sleep(1 + i)
                    else:
                        logger.error(f"[{getattr(self, 'name', 'Unknown')}] {op_name} failed after {retries} attempts.")

            if last_exc:
                raise last_exc

        return wrapper

    return decorator
