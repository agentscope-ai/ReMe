"""Shared ReMe protocol, workspace, and default values."""

REME_SERVICE_INFO = "REME_SERVICE_INFO"

REME_IPV4_LOOPBACK_HOST = "127.0.0.1"
REME_IPV6_LOOPBACK_HOST = "::1"

# The service and client defaults are used unless a host or port is configured.
REME_DEFAULT_HOST = REME_IPV4_LOOPBACK_HOST
REME_DEFAULT_PORT = 2333

# Wildcard address accepted when a service is explicitly configured to listen
# on every IPv4 interface.
REME_WILDCARD_BIND_HOST = "0.0.0.0"
REME_IPV6_WILDCARD_BIND_HOST = "::"

DEFAULT_MCP_PATH = "/mcp"
DEFAULT_MCP_SSE_PATH = "/sse"
OPENAPI_SCHEMA_PATH = "/openapi.json"
SSE_MEDIA_TYPE = "text/event-stream"
REME_SSE_DONE_MARKER = "[DONE]"
TRUSTED_SESSION_ID_KWARG = "_trusted_session_id"


def normalize_connect_host(host: str) -> str:
    """Return a local destination for a configured bind address."""
    if host == REME_WILDCARD_BIND_HOST:
        return REME_IPV4_LOOPBACK_HOST
    if host == REME_IPV6_WILDCARD_BIND_HOST:
        return REME_IPV6_LOOPBACK_HOST
    return host


# Fixed workspace and agent identifiers shared by their readers and writers.
SESSION_DIALOG_SUBDIR = "dialog"
CHAT_AGENT_NAME = "Remy"
PROACTIVE_MESSAGE_ID_PREFIX = "reme-proactive-"

WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    },
)

# CRUD steps: file IO limits and truncation marker (shared across CRUD steps).
DEFAULT_MAX_BYTES = 50 * 1024
MAX_FILE_READ_BYTES = 200 * 1024 * 1024
TRUNCATION_NOTICE_MARKER = "<<TRUNCATION_NOTICE>>"

# read_image step: oversized images above this threshold return path & metadata
# only (no base64) to keep LLM context budgets safe.
DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Background content-processing jobs skip files above this size. File watchers
# and catalogs still track them so deletes and later size reductions are seen.
DEFAULT_MAX_FILE_BYTES = 20 * 1024 * 1024

# Memory-tag generation and indexing defaults.
DEFAULT_MEMORY_TAG_KEY = "memory_tags"
DEFAULT_MAX_MEMORY_TAGS = 3
DEFAULT_MAX_MEMORY_TAG_LENGTH = 64
