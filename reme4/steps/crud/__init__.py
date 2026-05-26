"""File-level ops on vault_dir — both opaque-byte and text-content surfaces.

The package covers two related surfaces:

* **Opaque-byte ops** (don't care about file type): ``delete``,
  ``list``, ``move``, ``stat``.
* **Text-content ops** (markdown-aware; layered on the same path-
  resolution helpers in ``_file_io.py``): ``read``, ``write``,
  ``append``, ``edit``.

For frontmatter slice RUD (YAML structured-data semantics) see
``reme4.steps.frontmatter``. For vault ↔ local-fs bridge ops
(``download`` / ``upload`` / ``upload_resource``) see
``reme4.steps.transfer``.
"""

from .read import ReadStep
from .edit import EditStep
from .delete import DeleteStep
from .write import WriteStep
from .append import AppendStep
from .move import MoveStep
from .stat import StatStep
from .list import ListStep

__all__ = [
    "DeleteStep",
    "WriteStep",
    "AppendStep",
    "MoveStep",
    "StatStep",
    "ListStep",
    "ReadStep",
    "EditStep",
]
