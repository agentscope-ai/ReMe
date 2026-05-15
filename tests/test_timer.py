"""
This module provides a suite of tests to verify universal timer decorator functionality using loguru.
"""

import asyncio
import time

from loguru import logger

from reme.core.utils import timer


@timer
def test_sync_function(seconds: float) -> str:
    """Tests timing of a standard synchronous function."""
    time.sleep(seconds)
    return "sync done"


@timer
async def test_async_function(seconds: float) -> str:
    """Tests timing of an asynchronous function."""
    await asyncio.sleep(seconds)
    return "async done"


class TestMemberMethods:
    """Container class to test class method decoration."""

    @timer
    def test_sync_method(self, seconds: float) -> None:
        """Tests a synchronous instance method."""
        time.sleep(seconds)

    @timer
    async def test_async_method(self, seconds: float) -> None:
        """Tests an asynchronous instance method."""
        await asyncio.sleep(seconds)


def run_all_tests() -> None:
    """
    Manual test runner.
    Notice that the logs will now point to the line numbers below
    (where the function is actually called).
    """
    logger.info("Starting tests and verifying stack trace...")

    # 1. Test Sync Function
    test_sync_function(0.1)

    # 2. Test Async Function
    asyncio.run(test_async_function(0.1))

    # 3. Test Class Methods
    tester = TestMemberMethods()
    tester.test_sync_method(0.05)
    asyncio.run(tester.test_async_method(0.05))

    logger.success("All tests completed.")


if __name__ == "__main__":
    run_all_tests()
