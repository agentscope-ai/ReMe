"""Neo4j-backed file graph.

Property-graph mapping:

    (:File {path, st_mtime, title, description, tags, links_json,
            extra_json})
        -[:LINKS {idx, anchor, predicate}]->(:File)

``path`` is the unique key (constraint enforced on ``_start``).
Frontmatter goes into flat properties; arbitrary extras land in
``extra_json``. The full ``FileLink[]`` payload is also stored as
``links_json`` so we can recover the node losslessly even for links
whose target wasn't indexed at upsert time (will be linked later by
``_restore_inlinks``).

Adjacency policy: file_graph trusts ``FileLink.path`` directly — no
internal wikilink resolution. The parser pipeline (with the external
resolver) produces safe links where ``link.path`` is already a
vault-relative target path.

Conditional dependency: the ``neo4j`` driver is loaded lazily; the
import error fires at ``_start`` (boot), not at first call.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from .base_file_graph import BaseFileGraph
from ..component_registry import R
from ...schema import FileLink, FileNode
from ...schema.file_node import FileFrontMatter


_TYPED_FRONTMATTER_FIELDS = {"title", "description", "tags"}
_LINK_FIELDS = {"path", "anchor", "predicate"}


@R.register("neo4j")
class Neo4jFileGraph(BaseFileGraph):
    """Neo4j-backed file graph; trusts ``FileLink.path`` for adjacency.

    Connection params (constructor kwargs):
        uri:      bolt URL, e.g. ``bolt://localhost:7687``
        user:     auth user (default ``neo4j``)
        password: auth password
        database: target db name (default ``neo4j``)
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "neo4j",
        database: str = "neo4j",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._uri: str = uri
        self._user: str = user
        self._password: str = password
        self._database: str = database
        self._driver = None

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        await super()._start()
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as e:
            raise ImportError(
                "Neo4jFileGraph requires the neo4j driver. " "Install with `pip install neo4j`.",
            ) from e
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        async with self._session() as session:
            await session.run(
                "CREATE CONSTRAINT file_path_unique IF NOT EXISTS " "FOR (f:File) REQUIRE f.path IS UNIQUE",
            )
        self.logger.info(
            f"Neo4jFileGraph '{self.store_name}' connected at " f"{self._uri}/{self._database}",
        )

    async def _close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
        await super()._close()

    def _session(self):
        assert self._driver is not None, "Neo4jFileGraph not started"
        return self._driver.session(database=self._database)

    # -- Node CRUD ---------------------------------------------------------

    async def upsert_node(self, node: FileNode) -> None:
        props = self._node_props(node)
        # Trust link.path; only emit LINKS for links with non-empty path.
        link_payload = [
            {
                "idx": i,
                "anchor": link.anchor,
                "predicate": link.predicate,
                "target": link.path,
            }
            for i, link in enumerate(node.links)
            if link.path
        ]
        async with self._session() as session:
            await session.execute_write(
                self._upsert_node_tx,
                node.path,
                props,
                link_payload,
            )
        # Late-arriving target: restore in-links from other nodes whose
        # links resolve here. Separate session call (not in tx) so it
        # doesn't block the upsert ack.
        await self._restore_inlinks(node.path)

    @staticmethod
    async def _upsert_node_tx(tx, path, props, links):
        await tx.run(
            "MERGE (f:File {path: $path}) SET f += $props",
            path=path,
            props=props,
        )
        await tx.run(
            "MATCH (f:File {path: $path})-[r:LINKS]->() DELETE r",
            path=path,
        )
        # MERGE only when the target node exists. ``OPTIONAL MATCH`` +
        # ``WHERE t IS NOT NULL`` lets us batch this in one pass per link.
        for link in links:
            await tx.run(
                """
                MATCH (s:File {path: $src})
                OPTIONAL MATCH (t:File {path: $dst})
                WITH s, t WHERE t IS NOT NULL
                MERGE (s)-[r:LINKS {idx: $idx}]->(t)
                SET r.anchor = $anchor, r.predicate = $predicate
                """,
                src=path,
                dst=link["target"],
                idx=link["idx"],
                anchor=link["anchor"],
                predicate=link["predicate"],
            )

    async def _restore_inlinks(self, path: str) -> None:
        """Walk other nodes' links, add LINKS to ``path`` where ``link.path``
        matches. Uses ``links_json`` on each node — no resolution logic."""
        async with self._session() as session:
            rec = await session.run(
                """
                MATCH (f:File)
                WHERE f.path <> $path
                RETURN f.path AS p, f.links_json AS l
                """,
                path=path,
            )
            rows = [dict(r) async for r in rec]
        for row in rows:
            try:
                links = json.loads(row.get("l") or "[]")
            except json.JSONDecodeError:
                continue
            for i, link in enumerate(links):
                if not isinstance(link, dict) or link.get("path") != path:
                    continue
                async with self._session() as session:
                    await session.run(
                        """
                        MATCH (s:File {path: $src})
                        MATCH (t:File {path: $dst})
                        MERGE (s)-[r:LINKS {idx: $idx}]->(t)
                        SET r.anchor = $anchor, r.predicate = $predicate
                        """,
                        src=row["p"],
                        dst=path,
                        idx=i,
                        anchor=link.get("anchor"),
                        predicate=link.get("predicate"),
                    )

    async def delete_node(self, path: str) -> FileNode | None:
        node = await self.get_node(path)
        if node is None:
            return None
        async with self._session() as session:
            await session.run(
                "MATCH (f:File {path: $path}) DETACH DELETE f",
                path=path,
            )
        return node

    async def get_node(self, path: str) -> FileNode | None:
        async with self._session() as session:
            rec = await session.run(
                "MATCH (f:File {path: $path}) RETURN f LIMIT 1",
                path=path,
            )
            row = await rec.single()
        return self._row_to_node(row["f"]) if row else None

    # -- Link access -------------------------------------------------------

    async def get_outlinks(self, path: str) -> list[tuple[FileNode, FileLink]]:
        async with self._session() as session:
            rec = await session.run(
                """
                MATCH (s:File {path: $path})-[r:LINKS]->(t:File)
                RETURN t, r ORDER BY r.idx ASC
                """,
                path=path,
            )
            rows = [dict(row) async for row in rec]
        return [(self._row_to_node(r["t"]), self._rel_to_link(r["r"], r["t"])) for r in rows]

    async def get_inlinks(self, path: str) -> list[tuple[FileNode, FileLink]]:
        async with self._session() as session:
            rec = await session.run(
                """
                MATCH (s:File)-[r:LINKS]->(t:File {path: $path})
                RETURN s, r ORDER BY s.path ASC, r.idx ASC
                """,
                path=path,
            )
            rows = [dict(row) async for row in rec]
        # ``rel_to_link`` needs the target's path to populate FileLink.path.
        return [(self._row_to_node(r["s"]), self._rel_to_link(r["r"], target_path=path)) for r in rows]

    # -- Internal: row ↔ schema marshaling ---------------------------------

    @staticmethod
    def _node_props(node: FileNode) -> dict[str, Any]:
        fm = node.front_matter
        extras = dict(fm.__pydantic_extra__ or {})
        return {
            "path": node.path,
            "st_mtime": float(node.st_mtime),
            "title": fm.title or "",
            "description": fm.description or "",
            "tags": list(fm.tags or []),
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
            try:
                links.append(FileLink(**clean))
            except Exception:
                continue
        fm_kwargs: dict[str, Any] = {
            "title": d.get("title", "") or "",
            "description": d.get("description", "") or "",
            "tags": d.get("tags") or None,
        }
        fm_kwargs.update(
            {k: v for k, v in extras.items() if k not in _TYPED_FRONTMATTER_FIELDS},
        )
        return FileNode(
            path=d["path"],
            st_mtime=float(d.get("st_mtime", 0.0)),
            links=links,
            chunk_ids=[],
            front_matter=FileFrontMatter(**fm_kwargs),
        )

    @staticmethod
    def _rel_to_link(rel, target_path: Any = None) -> FileLink:
        """Reconstitute a ``FileLink`` from a Neo4j relationship.

        The relationship row carries ``anchor`` and ``predicate``; the
        target's ``path`` comes from the matched ``File`` node (passed
        in by callers that have it on hand). For ``get_outlinks`` it's
        the row's target node; for ``get_inlinks`` it's the path the
        caller queried for. Either way, FileLink.path is set so the
        link stays "safe by construction".
        """
        d = dict(rel)
        if hasattr(target_path, "get"):
            # neo4j Node passed in
            path = target_path.get("path", "")
        else:
            path = target_path or ""
        return FileLink(
            path=path,
            anchor=d.get("anchor"),
            predicate=d.get("predicate"),
        )
