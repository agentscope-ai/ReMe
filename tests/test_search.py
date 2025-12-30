import asyncio

from reme_ai.core import Application


async def async_main():
    """Test function for DashscopeSearchOp."""
    async with Application():
        from reme_ai.core.tool.search import DashscopeSearch

        op = DashscopeSearch()
        await op.call(query="藏格矿业的业务主要有哪几块？营收和利润的角度分析 雪球")
        print(op.output)


if __name__ == "__main__":
    asyncio.run(async_main())
