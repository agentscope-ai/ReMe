"""Utility modules."""

from .common_utils import (
    hash_text,
    execute_stream_task,
    mock_reme_server,
    call_action,
    call_and_check,
)
from .env_utils import load_env, parse_env_file
from .link_expansion import expand_links, render_expansion_lines
from .line_anchor import format_line_anchor, parse_line_anchor
from .logger_utils import get_logger
from .logo_utils import print_logo
from .service_utils import find_reme, locate_reme, precheck_start, cli_find_reme, running_service_config
from .similarity_utils import cosine_similarity, batch_cosine_similarity
from .token_utils import estimate_token_count
from .web_static import REME_WEB_STATIC_DIR, resolve_web_static_dir
from .agent_state_io import AsStateHandler
from .counter import (
    global_counter_add,
    global_counter_add_many,
    global_counter_get,
    global_counter_get_all,
    global_counter_inc,
)

__all__ = [
    "hash_text",
    "execute_stream_task",
    "mock_reme_server",
    "call_action",
    "call_and_check",
    "load_env",
    "parse_env_file",
    "expand_links",
    "render_expansion_lines",
    "format_line_anchor",
    "parse_line_anchor",
    "get_logger",
    "print_logo",
    "find_reme",
    "locate_reme",
    "precheck_start",
    "cli_find_reme",
    "running_service_config",
    "cosine_similarity",
    "batch_cosine_similarity",
    "estimate_token_count",
    "REME_WEB_STATIC_DIR",
    "resolve_web_static_dir",
    "AsStateHandler",
    "global_counter_add",
    "global_counter_add_many",
    "global_counter_get",
    "global_counter_get_all",
    "global_counter_inc",
]
