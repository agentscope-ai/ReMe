"""Service context for managing global application state.

This module provides a singleton service context that manages application-wide
configuration, registries, vector stores, and other shared resources.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from .base_context import BaseContext
from .registry import Registry
from ..enumeration import RegistryEnum
from ..schema import ServiceConfig
from ..utils import singleton


@singleton
class ServiceContext(BaseContext):
    """Singleton service context for global application state management.

    This class manages application-wide configuration including:
    - Service configuration and identification
    - Registry for models, operations, flows, and services
    - Vector stores
    - Thread pool executor
    - External MCP tool calls
    - Flow instances

    Attributes:
        service_id: Unique identifier for the service instance.
        service_config: Service configuration object.
        language: Language setting for the service.
        thread_pool: Thread pool executor for concurrent operations.
        vector_store_dict: Dictionary of vector store instances.
        external_mcp_tool_call_dict: Dictionary of external MCP tool calls.
        registry_dict: Dictionary of registries by type.
        flow_dict: Dictionary of flow instances.
    """

    def __init__(self, service_id: str | None = None, **kwargs):
        """Initialize ServiceContext with service ID.

        Args:
            service_id: Unique identifier for the service instance.
                Defaults to a random UUID hex string.
            **kwargs: Additional context data to store.
        """
        super().__init__(**kwargs)
        self.service_id: str = service_id or uuid.uuid4().hex

        self.service_config: ServiceConfig | None = None
        self.language: str = ""
        self.thread_pool: ThreadPoolExecutor | None = None
        self.vector_store_dict: dict = {}
        self.external_mcp_tool_call_dict: dict = {}
        self.registry_dict: Dict[RegistryEnum, Registry] = {v: Registry() for v in RegistryEnum.__members__.values()}
        self.flow_dict: dict = {}

    def register(self, name: str, register_type: RegistryEnum):
        """Register a model class by name and type.

        Args:
            name: Name to register the class under.
            register_type: Type of registry (e.g., LLM, EMBEDDING_MODEL).

        Returns:
            Decorator function for class registration.
        """
        return self.registry_dict[register_type].register(name=name)

    def register_embedding_model(self, name: str = ""):
        """Register an embedding model class.

        Args:
            name: Name to register the class under.

        Returns:
            Decorator function for class registration.
        """
        return self.register(name=name, register_type=RegistryEnum.EMBEDDING_MODEL)

    def register_llm(self, name: str = ""):
        """Register an LLM class.

        Args:
            name: Name to register the class under.

        Returns:
            Decorator function for class registration.
        """
        return self.register(name=name, register_type=RegistryEnum.LLM)

    def register_vector_store(self, name: str = ""):
        """Register a vector store class.

        Args:
            name: Name to register the class under.

        Returns:
            Decorator function for class registration.
        """
        return self.register(name=name, register_type=RegistryEnum.VECTOR_STORE)

    def register_op(self, name: str = ""):
        """Register an operation class.

        Args:
            name: Name to register the class under.

        Returns:
            Decorator function for class registration.
        """
        return self.register(name=name, register_type=RegistryEnum.OP)

    def register_flow(self, name: str = ""):
        """Register a flow class.

        Args:
            name: Name to register the class under.

        Returns:
            Decorator function for class registration.
        """
        return self.register(name=name, register_type=RegistryEnum.FLOW)

    def register_service(self, name: str = ""):
        """Register a service class.

        Args:
            name: Name to register the class under.

        Returns:
            Decorator function for class registration.
        """
        return self.register(name=name, register_type=RegistryEnum.SERVICE)

    def register_token_counter(self, name: str = ""):
        """Register a token counter class.

        Args:
            name: Name to register the class under.

        Returns:
            Decorator function for class registration.
        """
        return self.register(name=name, register_type=RegistryEnum.TOKEN_COUNTER)

    def get_model_class(self, name: str, register_type: RegistryEnum):
        """Get a registered model class by name and type.

        Args:
            name: Name of the registered class.
            register_type: Type of registry to search.

        Returns:
            The registered class.

        Raises:
            AssertionError: If the name is not found in the registry.
        """
        assert name in self.registry_dict[register_type], (
            f"name={name} not found in registry_dict.{register_type.value}! "
            f"supported names={self.registry_dict[register_type].keys()}"
        )

        return self.registry_dict[register_type][name]

    def get_embedding_model_class(self, name: str):
        """Get a registered embedding model class.

        Args:
            name: Name of the registered class.

        Returns:
            The registered embedding model class.

        Raises:
            AssertionError: If the name is not found in the registry.
        """
        return self.get_model_class(name, RegistryEnum.EMBEDDING_MODEL)

    def get_llm_class(self, name: str):
        """Get a registered LLM class.

        Args:
            name: Name of the registered class.

        Returns:
            The registered LLM class.

        Raises:
            AssertionError: If the name is not found in the registry.
        """
        return self.get_model_class(name, RegistryEnum.LLM)

    def get_vector_store_class(self, name: str):
        """Get a registered vector store class.

        Args:
            name: Name of the registered class.

        Returns:
            The registered vector store class.

        Raises:
            AssertionError: If the name is not found in the registry.
        """
        return self.get_model_class(name, RegistryEnum.VECTOR_STORE)

    def get_op_class(self, name: str):
        """Get a registered operation class.

        Args:
            name: Name of the registered class.

        Returns:
            The registered operation class.

        Raises:
            AssertionError: If the name is not found in the registry.
        """
        return self.get_model_class(name, RegistryEnum.OP)

    def get_flow_class(self, name: str):
        """Get a registered flow class.

        Args:
            name: Name of the registered class.

        Returns:
            The registered flow class.

        Raises:
            AssertionError: If the name is not found in the registry.
        """
        return self.get_model_class(name, RegistryEnum.FLOW)

    def get_service_class(self, name: str):
        """Get a registered service class.

        Args:
            name: Name of the registered class.

        Returns:
            The registered service class.

        Raises:
            AssertionError: If the name is not found in the registry.
        """
        return self.get_model_class(name, RegistryEnum.SERVICE)

    def get_token_counter_class(self, name: str):
        """Get a registered token counter class.

        Args:
            name: Name of the registered class.

        Returns:
            The registered token counter class.

        Raises:
            AssertionError: If the name is not found in the registry.
        """
        return self.get_model_class(name, RegistryEnum.TOKEN_COUNTER)

    def get_vector_store(self, name: str = "default"):
        """Get a vector store instance by name.

        Args:
            name: Name of the vector store instance. Defaults to "default".

        Returns:
            The vector store instance.
        """
        return self.vector_store_dict[name]

    def get_flow(self, name: str = "default"):
        """Get a flow instance by name.

        Args:
            name: Name of the flow instance. Defaults to "default".

        Returns:
            The flow instance.
        """
        return self.flow_dict[name]


C = ServiceContext()
