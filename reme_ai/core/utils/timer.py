import functools
import inspect
import logging
import time
from typing import Any, Callable, TypeVar, cast

# 配置日志（实际使用时通常在项目入口配置）
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# 定义泛型用于类型注解
F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    """
    计时器装饰器：支持同步、异步函数以及类成员方法。
    打印格式: ========== timer.{函数名}, time_cost={xxx}s ==========
    """

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.info(
                "========== timer.%s, time_cost=%.6fs ==========",
                func.__name__,
                duration,
                stacklevel=2
            )

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.info(
                "========== timer.%s, time_cost=%.6fs ==========",
                func.__name__,
                duration,
                stacklevel=2
            )

    # 判断是否为异步函数（包括 async def 定义的函数和类方法）
    if inspect.iscoroutinefunction(func):
        return cast(F, async_wrapper)
    return cast(F, sync_wrapper)
