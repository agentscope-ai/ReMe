"""Neo4j-backed file graph.

Property-graph mapping:

    Real node:    (:File {path, st_mtime, name, description,
                          chunk_ids, links_json, extra_json})
    Virtual node: (:File {path})  — placeholder created when something
                  links to a path that hasn't been upserted yet.

    Edge:         (:File)-[:LINKS {idx, anchor, predicate}]->(:File)

The ``links_json`` property doubles as the "is real" marker — its
presence means the node was upserted with a payload; its absence
means the node exists only because some edge points at it. This
mirrors ``NxFileGraph`` exactly: ``upsert_nodes`` promotes virtuals
in place, ``delete_nodes`` demotes back to virtual (or fully removes
if nothing points here), and ``get_outlinks`` excludes edges into
virtuals so the agent never sees dangling pointers.

``path`` is the unique key (constraint enforced on ``_start``).
Frontmatter goes into flat properties; arbitrary extras land in
``extra_json``. The full ``FileLink[]`` payload is also stored as
``links_json`` so ``rebuild_links`` can rebuild the relationship
graph from per-node payloads after backend repair / migration.

Adjacency policy: trusts ``FileLink.path`` directly — no internal
wikilink resolution. The parser pipeline (with the external
resolver) produces safe links where ``link.path`` is already a
target relative to the vault.

Conditional dependency: the ``neo4j`` driver loads lazily; the
import error fires at ``_start`` (boot), not at first call.
"""

import json
import os
from typing import Any

from .base_file_graph import BaseFileGraph
from ..component_registry import R
from ...enumeration import LinkScopeEnum
from ...schema import FileLink, FileNode
from ...schema.file_node import FileFrontMatter


_TYPED_FRONTMATTER_FIELDS = {"name", "description"}
_LINK_FIELDS = {"source_path", "target_path", "target_anchor", "predicate"}

# Properties that distinguish a "real" node from a virtual placeholder.
# Listed for the demote query (delete_nodes) so we can REMOVE them all.
_REAL_PROPS = (
    "st_mtime",
    "name",
    "description",
    "chunk_ids",
    "links_json",
    "extra_json",
)


@R.register("neo4j")
class Neo4jFileGraph(BaseFileGraph):
    """Neo4j-backed file graph; trusts ``FileLink.path`` for adjacency.

    Connection params (constructor kwargs):
        uri:      bolt URL, e.g. ``bolt://localhost:7687``
        user:     auth user (default ``neo4j``)
        password: auth password (required; falls back to ``NEO4J_PASSWORD`` env var)
        database: target db name (default ``neo4j``)
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str | None = None,
        database: str = "neo4j",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._uri: str = uri
        self._user: str = user
        self._password: str = password or os.environ.get("NEO4J_PASSWORD") or ""
        if not self._password:
            raise ValueError(
                "Neo4j password must be provided via the 'password' argument "
                "or the NEO4J_PASSWORD environment variable.",
            )
        self._database: str = database
        self._driver = None

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        await super()._start()
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as e:
            raise ImportError(
                "Neo4jFileGraph requires the neo4j driver. Install with `pip install neo4j`.",
            ) from e
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        async with self._session() as session:
            await session.run(
                "CREATE CONSTRAINT file_path_unique IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE",
            )
            real, virtual, edges = await self._counts(session)
        self.logger.info(
            f"Neo4jFileGraph '{self.name}' connected at "
            f"{self._uri}/{self._database}: "
            f"{real} nodes, {edges} edges, {virtual} virtual",
        )

    async def _close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
        await super()._close()

    def _session(self):
        assert self._driver is not None, "Neo4jFileGraph not started"
        return self._driver.session(database=self._database)

    @staticmethod
    async def _counts(session) -> tuple[int, int, int]:
        rec = await session.run(
            """
            MATCH (f:File)
            WITH count(CASE WHEN f.links_json IS NOT NULL THEN 1 END) AS real,
                 count(CASE WHEN f.links_json IS NULL THEN 1 END) AS virtual
            OPTIONAL MATCH ()-[r:LINKS]->()
            RETURN real, virtual, count(r) AS edges
            """,
        )
        row = await rec.single()
        if row is None:
            return 0, 0, 0
        return int(row["real"] or 0), int(row["virtual"] or 0), int(row["edges"] or 0)

    # -- Node CRUD ---------------------------------------------------------

    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
        """Upsert in one tx: SET props (promotes virtual to real), drop
        existing outgoing edges, re-emit edges (auto-creating virtual
        nodes for unindexed targets)."""
        if not nodes:
            return
        payload = [
            {
                "path": node.path,
                "props": self._node_props(node),
                "links": [
                    {
                        "idx": i,
                        "anchor": link.target_anchor,
                        "predicate": link.predicate,
                        "target": link.target_path,
                    }
                    for i, link in enumerate(node.links)
                    if link.target_path
                ],
            }
            for node in nodes
        ]
        async with self._session() as session:
            await session.execute_write(self._upsert_nodes_tx, payload)

    @staticmethod
    async def _upsert_nodes_tx(tx, payload):
        # 1. Upsert node props (promotes virtual → real where necessary).
        await tx.run(
            """
            UNWIND $items AS n
            MERGE (f:File {path: n.path})
            SET f += n.props
            """,
            items=payload,
        )
        # 2. Drop existing outgoing edges from these sources.
        await tx.run(
            """
            UNWIND $paths AS p
            MATCH (f:File {path: p})-[r:LINKS]->()
            DELETE r
            """,
            paths=[item["path"] for item in payload],
        )
        # 3. Re-emit edges; MERGE on target auto-creates virtual nodes
        # for unindexed targets.
        await tx.run(
            """
            UNWIND $items AS n
            MATCH (s:File {path: n.path})
            UNWIND n.links AS link
            MERGE (t:File {path: link.target})
            MERGE (s)-[r:LINKS {idx: link.idx}]->(t)
            SET r.anchor = link.anchor, r.predicate = link.predicate
            """,
            items=payload,
        )

    async def delete_nodes(self, paths: list[str]) -> None:
        """Demote real → virtual to preserve inbound visibility; fully
        remove the (now-virtual) node only if no edge points at it."""
        if not paths:
            return
        async with self._session() as session:
            await session.execute_write(self._delete_nodes_tx, list(paths))

    @staticmethod
    async def _delete_nodes_tx(tx, paths):
        # 1. Drop outgoing edges, then strip "real" properties (demote).
        # Building the REMOVE clause from _REAL_PROPS keeps the list of
        # properties in one place (top of module).
        remove_clause = ", ".join(f"f.{name}" for name in _REAL_PROPS)
        await tx.run(
            f"""
            UNWIND $paths AS p
            MATCH (f:File {{path: p}})
            OPTIONAL MATCH (f)-[r:LINKS]->()
            DELETE r
            WITH DISTINCT f
            REMOVE {remove_clause}
            """,
            paths=paths,
        )
        # 2. Garbage-collect: drop the virtual node entirely if nothing
        # points at it anymore.
        await tx.run(
            """
            UNWIND $paths AS p
            MATCH (f:File {path: p})
            WHERE f.links_json IS NULL AND NOT (f)<-[:LINKS]-()
            DELETE f
            """,
            paths=paths,
        )

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        """Return real nodes (virtual placeholders filtered).

        ``paths=None`` streams every real node ordered by path. An
        explicit ``[]`` returns ``[]`` without hitting the database.
        """
        if paths is not None and not paths:
            return []
        async with self._session() as session:
            if paths is None:
                rec = await session.run(
                    """
                    MATCH (f:File)
                    WHERE f.links_json IS NOT NULL
                    RETURN f
                    ORDER BY f.path ASC
                    """,
                )
            else:
                rec = await session.run(
                    """
                    UNWIND $paths AS p
                    MATCH (f:File {path: p})
                    WHERE f.links_json IS NOT NULL
                    RETURN f
                    """,
                    paths=list(paths),
                )
            rows = [row["f"] async for row in rec]
        return [self._row_to_node(row) for row in rows]

    async def rebuild_links(self) -> None:
        """Defensive full rebuild from each real node's ``links_json``.

        Three steps in one tx: drop all LINKS edges; drop all virtual
        nodes; re-emit edges from per-node link payloads (re-creating
        virtual targets as needed). Useful after manual repair or
        schema migration.
        """
        async with self._session() as session:
            rec = await session.run(
                """
                MATCH (f:File)
                WHERE f.links_json IS NOT NULL
                RETURN f.path AS p, f.links_json AS l
                """,
            )
            rows = [dict(r) async for r in rec]

        payload: list[dict] = []
        for row in rows:
            try:
                links = json.loads(row.get("l") or "[]")
            except json.JSONDecodeError:
                continue
            items = [
                {
                    "idx": i,
                    "anchor": link.get("target_anchor"),
                    "predicate": link.get("predicate"),
                    "target": link.get("target_path"),
                }
                for i, link in enumerate(links)
                if isinstance(link, dict) and link.get("target_path")
            ]
            payload.append({"path": row["p"], "links": items})

        async with self._session() as session:
            await session.execute_write(self._rebuild_links_tx, payload)

    @staticmethod
    async def _rebuild_links_tx(tx, payload):
        # 1. Wipe all edges and all virtual nodes.
        await tx.run("MATCH ()-[r:LINKS]->() DELETE r")
        await tx.run("MATCH (f:File) WHERE f.links_json IS NULL DELETE f")
        if not payload:
            return
        # 2. Re-emit edges; virtual targets reappear via MERGE.
        await tx.run(
            """
            UNWIND $items AS n
            MATCH (s:File {path: n.path})
            UNWIND n.links AS link
            MERGE (t:File {path: link.target})
            MERGE (s)-[r:LINKS {idx: link.idx}]->(t)
            SET r.anchor = link.anchor, r.predicate = link.predicate
            """,
            items=payload,
        )

    async def clear(self):
        """Remove every node and edge in the configured database."""
        async with self._session() as session:
            await session.run("MATCH (f:File) DETACH DELETE f")

    # -- Link access -------------------------------------------------------

    async def get_outlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        """Outgoing links from ``path``. Source must be real; ``scope``
        selects which targets to surface (REAL / VIRTUAL / ALL).
        """
        target_filter = _neo4j_scope_filter("t", scope)
        async with self._session() as session:
            rec = await session.run(
                f"""
                MATCH (s:File {{path: $path}})
                WHERE s.links_json IS NOT NULL
                MATCH (s)-[r:LINKS]->(t:File)
                WHERE 1=1 {target_filter}
                RETURN t.path AS target, r.anchor AS anchor,
                       r.predicate AS predicate, r.idx AS idx
                ORDER BY r.idx ASC
                """,
                path=path,
            )
            rows = [dict(row) async for row in rec]
        return [
            FileLink(
                source_path=path,
                target_path=row["target"],
                target_anchor=row.get("anchor"),
                predicate=row.get("predicate"),
            )
            for row in rows
        ]

    async def get_inlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        """Incoming links to ``path`` from real sources.

        ``scope`` selects whether ``path`` itself must be real / virtual /
        either; sources are always real (virtual nodes have no outgoing
        edges to begin with).
        """
        target_filter = _neo4j_scope_filter("t", scope)
        async with self._session() as session:
            rec = await session.run(
                f"""
                MATCH (t:File {{path: $path}})
                WHERE 1=1 {target_filter}
                MATCH (s:File)-[r:LINKS]->(t)
                WHERE s.links_json IS NOT NULL
                RETURN r.anchor AS anchor, r.predicate AS predicate,
                       r.idx AS idx, s.path AS source
                ORDER BY s.path ASC, r.idx ASC
                """,
                path=path,
            )
            rows = [dict(row) async for row in rec]
        return [
            FileLink(
                source_path=row["source"],
                target_path=path,
                target_anchor=row.get("anchor"),
                predicate=row.get("predicate"),
            )
            for row in rows
        ]

    # -- Internal: row ↔ schema marshaling ---------------------------------

    @staticmethod
    def _node_props(node: FileNode) -> dict[str, Any]:
        fm = node.front_matter
        extras = dict(fm.__pydantic_extra__ or {})
        return {
            "path": node.path,
            "st_mtime": float(node.st_mtime),
            "name": fm.name or "",
            "description": fm.description or "",
            "chunk_ids": list(node.chunk_ids or []),
            "links_json": json.dumps(
                [link.model_dump(exclude_none=True) for link in node.links],
                ensure_ascii=False,
            ),
            "extra_json": json.dumps(extras, ensure_ascii=False, sort_keys=True),
        }

    @staticmethod
    def _row_to_node(row) -> FileNode:
        d = dict(row)
        try:
            extras = json.loads(d.get("extra_json") or "{}")
        except json.JSONDecodeError:
            extras = {}
        try:
            links_raw = json.loads(d.get("links_json") or "[]")
        except json.JSONDecodeError:
            links_raw = []
        links: list[FileLink] = []
        for link in links_raw:
            if not isinstance(link, dict):
                continue
            # Defensive: strip any keys the schema doesn't recognise
            # (e.g. legacy fields from prior schema versions).
            clean = {k: v for k, v in link.items() if k in _LINK_FIELDS}
            # Ensure source_path is populated — older payloads (or
            # links written before the schema split) only carry the
            # target side; default to the owning node's path.
            clean.setdefault("source_path", d["path"])
            if not clean.get("target_path"):
                continue
            try:
                links.append(FileLink(**clean))
            except Exception:
                continue
        fm_kwargs: dict[str, Any] = {
            "name": d.get("name", "") or "",
            "description": d.get("description", "") or "",
        }
        fm_kwargs.update(
            {k: v for k, v in extras.items() if k not in _TYPED_FRONTMATTER_FIELDS},
        )
        return FileNode(
            path=d["path"],
            st_mtime=float(d.get("st_mtime", 0.0)),
            links=links,
            chunk_ids=[str(c) for c in (d.get("chunk_ids") or [])],
            front_matter=FileFrontMatter(**fm_kwargs),
        )


def _neo4j_scope_filter(node_var: str, scope: LinkScopeEnum) -> str:
    """Cypher predicate that restricts ``node_var`` to the requested scope.

    Encodes the same convention used everywhere in this backend:
    ``links_json IS NOT NULL`` ⇔ real node; ``IS NULL`` ⇔ virtual
    placeholder. ``ALL`` returns an empty fragment so callers can
    splice it after ``WHERE 1=1``.
    """
    if scope is LinkScopeEnum.REAL:
        return f"AND {node_var}.links_json IS NOT NULL"
    if scope is LinkScopeEnum.VIRTUAL:
        return f"AND {node_var}.links_json IS NULL"
    return ""
