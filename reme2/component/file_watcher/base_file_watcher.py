from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ..file_parser import BaseFileParser


class BaseFileWatcher(BaseComponent):
    component_type = ComponentEnum.FILE_WATCHER

    def __init__(
            self,
            watch_path: list[str],
            recursive: bool = False,
            file_parser: str = "default",
            file_store: str = "default",
            debounce: int = 2000,
            poll_delay_ms: int = 2000,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.watch_path: list[str] = watch_path
        self.recursive: bool = recursive
        self.file_parser_name: str = file_parser
        self.file_store: str = file_store
        self.debounce: int = debounce
        self.poll_delay_ms: int = poll_delay_ms

        self.file_parser: BaseFileParser | None = None

    async def _start(self) -> None:
        await super()._start()
        if self.app_context is not None:
            file_parser_dict = self.app_context.components.get(ComponentEnum.FILE_PARSER, {})
            if self.file_parser not in file_parser_dict:
                raise ValueError(f"File parser {self.file_parser} not found")
            self.file_parser = file_parser_dict[self.file_parser]

    async def _close(self) -> None:
        await super()._close()