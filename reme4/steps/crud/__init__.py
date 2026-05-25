"""File-level ops on vault_dir — both opaque-byte and text-content surfaces.

The package covers two related surfaces:

* **Opaque-byte ops** (don't care about file type): ``delete``,
  ``download``, ``list``, ``move``, ``stat``, ``upload``,
  ``upload_resource``.
* **Text-content ops** (markdown-aware; layered on the same path-
  resolution helpers in ``_file_io.py``): ``read``, ``write``,
  ``append``, ``edit``.

For frontmatter slice RUD (YAML structured-data semantics) see
``reme4.steps.frontmatter``.
"""

from .read import ReadStep
from .edit import EditStep
from .delete import DeleteStep
from .write import WriteStep
from .append import AppendStep
from .move import MoveStep
from .stat import StatStep
from .download import DownloadStep
from .list import ListStep
from .upload import UploadStep
from .upload_resource import UploadResourceStep

__all__ = [
    "DeleteStep",
    "WriteStep",
    "AppendStep",
    "MoveStep",
    "StatStep",
    "DownloadStep",
    "ListStep",
    "UploadStep",
    "UploadResourceStep",
    "ReadStep",
    "EditStep",
]
