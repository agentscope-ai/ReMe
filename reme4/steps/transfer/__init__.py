"""Bridge ops between vault_dir and the local filesystem.

Unlike ``crud`` (single-resource ops within the vault), these steps
cross two domains: vault ↔ host fs.

* ``download`` — vault → local fs (export; vault is read-only).
* ``upload`` — local fs → vault (raw copy to a caller-supplied path).
* ``upload_resource`` — local fs → ``resource/<YYYY-MM-DD>/`` (passive
  ingest with provenance metadata + day-view rendering).
"""

from .download import DownloadStep
from .upload import UploadStep
from .upload_resource import UploadResourceStep

__all__ = [
    "DownloadStep",
    "UploadStep",
    "UploadResourceStep",
]
