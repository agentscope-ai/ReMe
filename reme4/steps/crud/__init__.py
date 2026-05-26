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

from .append import AppendStep
from .delete import DeleteStep
from .download import DownloadStep
from .edit import EditStep
from .list import ListStep
from .move import MoveStep
from .read import ReadStep
from .read_image import ReadImageStep
from .stat import StatStep
from .upload import UploadStep
from .upload_resource import UploadResourceStep
from .write import WriteStep

__all__ = [
    "AppendStep",
    "DeleteStep",
    "DownloadStep",
    "EditStep",
    "ListStep",
    "MoveStep",
    "ReadImageStep",
    "ReadStep",
    "StatStep",
    "UploadResourceStep",
    "UploadStep",
    "WriteStep",
]
