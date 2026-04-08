from reme_cli.component import BaseComponent


class Application(BaseComponent):
    """Application component for managing the main application."""

    def __init__(self) -> None:
        super().__init__()
        ...

    async def start(self) -> None:
        """Start the application."""
        # 初始化llm formater
        #
        pass

    async def close(self) -> None:
        """Close the application."""
        pass

