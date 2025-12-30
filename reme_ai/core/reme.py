import sys

from .application import Application
from .config import ReMeConfigParser


class ReMe(Application):
    def __init__(
        self,
        *args,
        llm_api_key: str = None,
        llm_api_base: str = None,
        embedding_api_key: str = None,
        embedding_api_base: str = None,
        llm: dict | None = None,
        embedding_model: dict | None = None,
        vector_store: dict | None = None,
        token_counter: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            service_config=None,
            parser=ReMeConfigParser,
            config_path=None,
            load_default_config=True,
            **kwargs,
        )

        self.service_config


class ReMeApp(Application):
    def __init__(
        self,
        *args,
        llm_api_key: str = None,
        llm_api_base: str = None,
        embedding_api_key: str = None,
        embedding_api_base: str = None,
        config_path: str = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            service_config=None,
            parser=ReMeConfigParser,
            config_path=config_path,
            load_default_config=True,
            **kwargs,
        )

    async def async_execute(self, name: str, **kwargs) -> dict:
        return (await self.execute_flow(name=name, **kwargs)).model_dump()


def main():
    with ReMeApp(*sys.argv[1:]) as app:
        app.run_service()


if __name__ == "__main__":
    main()
