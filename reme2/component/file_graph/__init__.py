"""File graph module.

Backend-agnostic graph engine that owns the wikilink graph (nodes +
typed edges + traversal queries). Sister to ``file_store``, which owns
chunks and projections (vector / FTS).

Two backends ship today:
    - ``LocalFileGraph`` (``@R.register("local")``) — networkx
      ``MultiDiGraph`` + JSONL persistence. Default.
    - ``Neo4jFileGraph`` (``@R.register("neo4j")``) — property-graph
      backed by Neo4j. Requires the ``neo4j`` driver.
"""

from .base_file_graph import BaseFileGraph
from .local_file_graph import LocalFileGraph
from .neo4j_file_graph import Neo4jFileGraph

__all__ = [
    "BaseFileGraph",
    "LocalFileGraph",
    "Neo4jFileGraph",
]
