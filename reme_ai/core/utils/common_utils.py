import asyncio


def run_coro_safely(coro):
    try:
        loop = asyncio.get_running_loop()

    except RuntimeError:
        return asyncio.run(coro)

    else:
        return loop.create_task(coro)
