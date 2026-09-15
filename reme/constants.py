"""Constants"""

REME_SERVICE_INFO = "REME_SERVICE_INFO"

# Loopback address used by services unless remote access is explicitly enabled.
REME_DEFAULT_BIND_HOST = "127.0.0.1"

# Wildcard address accepted when a service is explicitly configured to listen
# on every IPv4 interface.
REME_WILDCARD_BIND_HOST = "0.0.0.0"

# Loopback address used by clients when no remote service is configured.
REME_DEFAULT_CONNECT_HOST = "127.0.0.1"

# Backward-compatible alias for callers that historically used the single
# default host as a client destination.
REME_DEFAULT_HOST = REME_DEFAULT_CONNECT_HOST

REME_DEFAULT_PORT = 2333

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
