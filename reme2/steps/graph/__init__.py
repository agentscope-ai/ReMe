"""Graph steps — wikilink relationship exploration.

Single Step:

    graph:traverse  — BFS from a seed file over wikilink edges.

Direction vocabulary matches the obsidian convention
(``forward`` / ``backward`` / ``both``); the engine vocab
(``out`` / ``in`` / ``both``) is accepted as alias. Inbound
traversal walks the vault to reconstruct source paths since the
file_graph contract's ``get_inlinks`` only returns target-shaped
FileLinks.
"""

from . import traverse  # noqa: F401  -- @R.register("graph:traverse")
