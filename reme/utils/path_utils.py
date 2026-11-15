"""Workspace-relative path identity shared by every index consumer."""


def canonical_index_path(path: str) -> str:
    """Return the workspace-relative POSIX spelling used as index identity.

    Indexes persisted by older Windows builds store native separators in
    ``FileNode.path`` and ``FileChunk.path``, while the tag index, the search
    filter and freshly indexed files use POSIX separators. Every consumer
    compares these identities, so the same canonical form is applied when
    loading, when writing and to query arguments; otherwise the two spellings
    address different files.

    Only separators are rewritten. Rejecting absolute or escaping paths stays
    with the caller that owns the containment rule.
    """
    return path.replace("\\", "/")
