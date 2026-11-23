"""Shared Markdown parsing for callers that use mistletoe."""

from threading import Lock

# Both Document and MarkdownRenderer use process-wide parser state.
MARKDOWN_PARSE_LOCK = Lock()


def markdown_link_targets(content: str) -> set[str]:
    """Collect Markdown links and image targets outside code spans and blocks."""
    if not content:
        return set()
    from mistletoe.block_token import Document
    from mistletoe.span_token import Image, Link

    with MARKDOWN_PARSE_LOCK:
        document = Document(content.replace("\r\n", "\n"))

    targets = set()
    nodes = [document]
    while nodes:
        node = nodes.pop()
        if isinstance(node, Link):
            targets.add(node.target)
        elif isinstance(node, Image):
            targets.add(node.src)
        nodes.extend(node.children or ())
    return targets
