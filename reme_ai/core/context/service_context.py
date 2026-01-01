"""Module for managing global service configurations and component registries via a singleton context."""

from concurrent.futures import ThreadPoolExecutor

from .base_context import BaseContext
from .registry import Registry
from ..enumeration import RegistryEnum
from ..schema import ServiceConfig
from ..utils import singleton


@singleton
class ServiceContext(BaseContext):
    """A singleton container for global application state, thread pools, and component registries."""

    def __init__(self, **kwargs):
        """Initialize the global context with configuration objects and specialized registries."""
        super().__init__(**kwargs)

        self.service_config: ServiceConfig | None = None
        self.language: str = ""
        self.thread_pool: ThreadPoolExecutor | None = None
        self.vector_store_dict: dict[str, dict] = {}
        self.mcp_server_tool_call_mapping: dict = {}
        # Initialize a registry for every category defined in RegistryEnum
        self.registry_dict: dict[RegistryEnum, Registry] = {v: Registry() for v in RegistryEnum.__members__.values()}
        self.flow_dict: dict = {}

    def register(self, name: str, register_type: RegistryEnum):
        """Return a decorator to register a component within a specific registry category."""
        return self.registry_dict[register_type].register(name=name)

    def register_llm(self, name: str = ""):
        """Register a Large Language Model class."""
        return self.register(name=name, register_type=RegistryEnum.LLM)

    def register_embedding_model(self, name: str = ""):
        """Register an embedding model class."""
        return self.register(name=name, register_type=RegistryEnum.EMBEDDING_MODEL)

    def register_vector_store(self, name: str = ""):
        """Register a vector store implementation class."""
        return self.register(name=name, register_type=RegistryEnum.VECTOR_STORE)

    def register_op(self, name: str = ""):
        """Register an operation (Op) class."""
        return self.register(name=name, register_type=RegistryEnum.OP)

    def register_flow(self, name: str = ""):
        """Register a workflow or logic flow class."""
        return self.register(name=name, register_type=RegistryEnum.FLOW)

    def register_service(self, name: str = ""):
        """Register a backend service class."""
        return self.register(name=name, register_type=RegistryEnum.SERVICE)

    def register_token_counter(self, name: str = ""):
        """Register a token counting utility class."""
        return self.register(name=name, register_type=RegistryEnum.TOKEN_COUNTER)

    def get_model_class(self, name: str, register_type: RegistryEnum):
        """Retrieve a registered class by name from a specific registry category."""
        assert name in self.registry_dict[register_type], f"{name} not in registry_dict[{register_type}]"
        return self.registry_dict[register_type][name]

    def get_embedding_model_class(self, name: str):
        """Get the embedding model class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.EMBEDDING_MODEL)

    def get_llm_class(self, name: str):
        """Get the LLM class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.LLM)

    def get_vector_store_class(self, name: str):
        """Get the vector store class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.VECTOR_STORE)

    def get_op_class(self, name: str):
        """Get the operation class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.OP)

    def get_flow_class(self, name: str):
        """Get the flow class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.FLOW)

    def get_service_class(self, name: str):
        """Get the service class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.SERVICE)

    def get_token_counter_class(self, name: str):
        """Get the token counter class registered under the given name."""
        return self.get_model_class(name, RegistryEnum.TOKEN_COUNTER)

    def get_vector_store(self, name: str):
        """Retrieve a specific vector store instance by name."""
        return self.vector_store_dict[name]

    def get_flow(self, name: str):
        """Retrieve a specific flow instance by name."""
        return self.flow_dict[name]


# Export a global instance for easy access across the application
C = ServiceContext()
