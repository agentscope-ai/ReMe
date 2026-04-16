"""Constants"""

REME_SERVICE_INFO = "REME_SERVICE_INFO"

REME_DEFAULT_HOST = "127.0.0.1"

REME_DEFAULT_PORT = 2333

# Default truncation limit for text output
DEFAULT_MAX_BYTES = 50 * 1024

# Maximum file size to read into memory (1GB)
MAX_FILE_READ_BYTES = 1024 * 1024 * 1024

# Marker prepended to every truncation notice.
# Format:
#   <<<TRUNCATED>>>
#   The output above was truncated.
#   The full content is saved to the file and contains Z lines in total.
#   This excerpt starts at line X and covers the next N bytes.
#   If the current content is not enough, call `read_file` with file_path=<path> start_line=Y to read more.
#
# Split output on this marker to recover the original (untruncated) portion:
#   original = output.split(TRUNCATION_NOTICE_MARKER)[0]
TRUNCATION_NOTICE_MARKER = "<<<TRUNCATED>>>"
