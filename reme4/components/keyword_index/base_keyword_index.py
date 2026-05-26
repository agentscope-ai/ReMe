from abc import abstractmethod

from ..base_component import BaseComponent
from ..tokenizer import BaseTokenizer
from ...enumeration import ComponentEnum


class BaseKeywordIndex(BaseComponent):
    """关键词索引基类：定义增、删、查、清的统一接口，由具体实现（如 BM25）继承。"""

    component_type = ComponentEnum.KEYWORD_INDEX

    def __init__(self, tokenizer: str = "default", **kwargs):
        super().__init__(**kwargs)
        from ..tokenizer import RegexTokenizer

        # 绑定分词器，未显式指定时回落到 RegexTokenizer
        self.tokenizer = self.bind(tokenizer, BaseTokenizer, default_factory=RegexTokenizer)
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)

    async def _start(self) -> None:
        await self.load()

    async def _close(self) -> None:
        await self.dump()

    def _tokenize(self, text: str) -> list[str]:
        """对单段文本调用分词器，返回 token 列表。"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call start() first.")
        return self.tokenizer.tokenize([text])[0]

    @abstractmethod
    async def add_docs(self, docs_dict: dict[str, str]) -> None: ...

    @abstractmethod
    async def delete_docs(self, doc_ids: list[str]) -> None: ...

    @abstractmethod
    async def retrieve(self, query: str, limit: int = 3) -> dict[str, float]: ...

    @abstractmethod
    async def clear(self) -> None: ...

    async def reset_index(self, docs_dict: dict[str, str]) -> None:
        """清空索引后重新构建，并立即落盘。"""
        await self.clear()
        await self.add_docs(docs_dict)
        await self.dump()

    async def optimize_index(self) -> None:
        """对索引进行物理压缩或重建；基类默认无操作，由子类按需重载。"""
        pass
