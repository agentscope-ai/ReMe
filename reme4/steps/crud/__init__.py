"""File-level ops on vault_dir — both opaque-byte and text-content surfaces.

The package covers two related surfaces:

* **Opaque-byte ops** (don't care about file type): ``delete``,
  ``download``, ``list``, ``move``, ``stat``, ``upload``.
* **Text-content ops** (markdown-aware; layered on the same path-
  resolution helpers in ``_file_io.py``): ``read``, ``write``,
  ``append``, ``edit``.

For frontmatter slice RUD (YAML structured-data semantics) see
``reme4.steps.frontmatter``.
"""

# Module names list/stat mirror their tool names.
# pylint: disable=redefined-builtin

from . import delete  # noqa: F401  -- @R.register("delete_step")
from . import download  # noqa: F401  -- @R.register("download_step")
from . import list  # noqa: F401  -- @R.register("list_step")
from . import move  # noqa: F401  -- @R.register("move_step")
from . import stat  # noqa: F401  -- @R.register("stat_step")
from . import upload  # noqa: F401  -- @R.register("upload_step")
from . import read  # noqa: F401  -- @R.register("read_step")
from . import write  # noqa: F401  -- @R.register("write_step")
from . import append  # noqa: F401  -- @R.register("append_step")
from . import edit  # noqa: F401  -- @R.register("edit_step")

from .read import ReadStep
from .edit import EditStep

__all__ = [
    "ReadStep",
    "EditStep",
]
