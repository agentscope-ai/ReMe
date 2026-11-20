"""Components and their shared interfaces."""

import sys

from . import agent_wrapper
from . import anthropic_relay
from . import as_llm
from . import as_decision_model
from . import asr
from . import client
from . import codex_app_server
from .codex_app_server import BaseCodexAppServer, SdkCodexAppServer
from . import as_embedding
from . import embedding_store
from . import agent_inbox
from .agent_inbox import BaseAgentInbox, LocalAgentInbox, AgentEventInbox
from . import file_catalog
from . import file_graph
from . import file_chunker
from . import file_store
from . import job
from . import keyword_index
from . import outbound_proxy
from . import agent_runtime
from .agent_runtime import BaseAgentRuntime, LocalAgentRuntime
from . import service
from . import tag_index
from . import tokenizer
from . import tts
from . import base as base_component
from .base import BaseComponent, ComponentMixin
from ..core import context as application_context
from ..core import registry as component_registry
from ..core.context import ApplicationContext, RuntimeContext
from ..core.registry import ComponentRegistry, R, create_application_registry
from ..steps import prompt as prompt_handler
from ..steps.prompt import PromptHandler

# Keep published component imports pointed at their canonical modules.
runtime_context = application_context
for _name, _module in {
    "application_context": application_context,
    "base_component": base_component,
    "component_registry": component_registry,
    "prompt_handler": prompt_handler,
    "runtime_context": runtime_context,
}.items():
    sys.modules[f"{__name__}.{_name}"] = _module

__all__ = [
    "ApplicationContext",
    "BaseComponent",
    "ComponentMixin",
    "ComponentRegistry",
    "BaseCodexAppServer",
    "SdkCodexAppServer",
    "BaseAgentInbox",
    "LocalAgentInbox",
    "AgentEventInbox",
    "BaseAgentRuntime",
    "LocalAgentRuntime",
    "R",
    "create_application_registry",
    "PromptHandler",
    "RuntimeContext",
    # base components
    "agent_wrapper",
    "agent_runtime",
    "anthropic_relay",
    "codex_app_server",
    "agent_inbox",
    "as_llm",
    "as_decision_model",
    "asr",
    "client",
    "as_embedding",
    "embedding_store",
    "file_catalog",
    "file_graph",
    "file_chunker",
    "file_store",
    "job",
    "keyword_index",
    "outbound_proxy",
    "service",
    "tag_index",
    "tokenizer",
    "tts",
]
