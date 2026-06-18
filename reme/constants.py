"""Constants"""

REME_SERVICE_INFO = "REME_SERVICE_INFO"

REME_DEFAULT_HOST = "127.0.0.1"

REME_DEFAULT_PORT = 2333

# CRUD steps: file IO limits and truncation marker (shared across CRUD steps).
DEFAULT_MAX_BYTES = 50 * 1024
MAX_FILE_READ_BYTES = 200 * 1024 * 1024
TRUNCATION_NOTICE_MARKER = "<<TRUNCATION_NOTICE>>"

# read_image step: oversized images above this threshold return path & metadata
# only (no base64) to keep LLM context budgets safe.
DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024
