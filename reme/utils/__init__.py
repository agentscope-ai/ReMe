"""Shared utility imports."""

from .counter import (
    global_counter_add,
    global_counter_add_many,
    global_counter_get,
    global_counter_get_all,
    global_counter_inc,
)
from .documents.line_anchor import format_line_anchor, parse_line_anchor
from .documents.link_expansion import expand_links, render_expansion_lines
from .env_utils import load_env, parse_env_file
from .files.paths import canonical_index_path
from .hashing import hash_text
from .logger_utils import get_logger
from .qwen_token_tools import QwenTokenTools, estimate_token_count
from .similarity_utils import batch_cosine_similarity, cosine_similarity

__all__ = [
    "hash_text",
    "load_env",
    "parse_env_file",
    "expand_links",
    "render_expansion_lines",
    "format_line_anchor",
    "parse_line_anchor",
    "get_logger",
    "cosine_similarity",
    "batch_cosine_similarity",
    "canonical_index_path",
    "estimate_token_count",
    "QwenTokenTools",
    "global_counter_add",
    "global_counter_add_many",
    "global_counter_get",
    "global_counter_get_all",
    "global_counter_inc",
]
