from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from .base_context import BaseContext
from .registry import Registry
from ..enumeration import RegistryEnum
from ..schema import ServiceConfig
from ..utils import singleton


@singleton
class ServiceContext(BaseContext):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.service_config: ServiceConfig | None = None
        self.language: str = ""
        self.thread_pool: ThreadPoolExecutor | None = None
        self.vector_store_dict: dict = {}
        self.external_mcp_tool_call_dict: dict = {}
        self.registry_dict: Dict[RegistryEnum, Registry] = {v: Registry() for v in RegistryEnum.__members__.values()}
        self.flow_dict: dict = {}

    def register(self, name: str, register_type: RegistryEnum):
        return self.registry_dict[register_type].register(name=name)

    def register_llm(self, name: str = ""):
        return self.register(name=name, register_type=RegistryEnum.LLM)

    def register_embedding_model(self, name: str = ""):
        return self.register(name=name, register_type=RegistryEnum.EMBEDDING_MODEL)

    def register_vector_store(self, name: str = ""):
        return self.register(name=name, register_type=RegistryEnum.VECTOR_STORE)

    def register_op(self, name: str = ""):
        return self.register(name=name, register_type=RegistryEnum.OP)

    def register_flow(self, name: str = ""):
        return self.register(name=name, register_type=RegistryEnum.FLOW)

    def register_service(self, name: str = ""):
        return self.register(name=name, register_type=RegistryEnum.SERVICE)

    def register_token_counter(self, name: str = ""):
        return self.register(name=name, register_type=RegistryEnum.TOKEN_COUNTER)

    def get_model_class(self, name: str, register_type: RegistryEnum):
        assert name in self.registry_dict[register_type], f"{name} not in registry_dict[{register_type}]"
        return self.registry_dict[register_type][name]

    def get_embedding_model_class(self, name: str):
        return self.get_model_class(name, RegistryEnum.EMBEDDING_MODEL)

    def get_llm_class(self, name: str):
        return self.get_model_class(name, RegistryEnum.LLM)

    def get_vector_store_class(self, name: str):
        return self.get_model_class(name, RegistryEnum.VECTOR_STORE)

    def get_op_class(self, name: str):
        return self.get_model_class(name, RegistryEnum.OP)

    def get_flow_class(self, name: str):
        return self.get_model_class(name, RegistryEnum.FLOW)

    def get_service_class(self, name: str):
        return self.get_model_class(name, RegistryEnum.SERVICE)

    def get_token_counter_class(self, name: str):
        return self.get_model_class(name, RegistryEnum.TOKEN_COUNTER)

    def get_vector_store(self, name: str):
        return self.vector_store_dict[name]

    def get_flow(self, name: str):
        return self.flow_dict[name]


C = ServiceContext()
