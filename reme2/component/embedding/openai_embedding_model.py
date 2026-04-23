"""OpenAI-compatible async embedding model."""

from openai import AsyncOpenAI

from .base_embedding_model import BaseEmbeddingModel
from ..component_registry import R


@R.register("openai")
class OpenAIEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client: AsyncOpenAI | None = None

    async def _start(self) -> None:
        self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, **self.kwargs)
        await super()._start()

    async def _close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
        await super()._close()

    async def _get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float] | None]:
        if self._client is None:
            raise RuntimeError("Client not initialized. Call _start() first.")

        create_kwargs: dict = {"model": self.model_name, "input": input_text, **kwargs}
        if self.pass_dimensions:
            create_kwargs["dimensions"] = self.dimensions

        completion = await self._client.embeddings.create(**create_kwargs)

        result: list[list[float] | None] = [None] * len(input_text)
        for emb in completion.data:
            vec = getattr(emb, "embedding", None) or getattr(emb, "dense_embedding", None)
            if 0 <= emb.index < len(input_text):
                if vec is not None:
                    result[emb.index] = list(vec)
                else:
                    self.logger.warning(f"Empty embedding for index {emb.index}")
            else:
                self.logger.warning(f"Invalid index {emb.index} for input length {len(input_text)}")

        return result
