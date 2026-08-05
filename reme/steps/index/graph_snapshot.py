"""Return the digest graph, rooted by its three memory categories."""

from ..base_step import BaseStep
from ...components import R
from ...enumeration import DreamBucketEnum
from ...schema import GraphSnapshot, GraphSnapshotEdge, GraphSnapshotNode

_CATEGORY_BUCKETS = (
    DreamBucketEnum.WIKI,
    DreamBucketEnum.PERSONAL,
    DreamBucketEnum.PROCEDURE,
)


@R.register("graph_snapshot_step")
class GraphSnapshotStep(BaseStep):
    """Build the frontend digest graph through the file-store contract.

    Category nodes connect to every indexed Markdown file in their bucket.
    Digest files retain wikilinks to other digest files and to daily notes. Daily
    notes are leaves: their own outgoing links are intentionally not returned.
    """

    async def execute(self):
        assert self.context is not None
        indexed_nodes = await self.file_store.get_nodes()
        node_by_path = {node.path: node for node in indexed_nodes if node.path.lower().endswith(".md")}

        digest_dir = str(self.config_value("digest_dir")).strip("/")
        daily_dir = str(self.config_value("daily_dir")).strip("/")
        category_paths = {
            bucket: f"{digest_dir}/{bucket.value}" if digest_dir else bucket.value for bucket in _CATEGORY_BUCKETS
        }
        daily_prefix = f"{daily_dir}/" if daily_dir else ""

        digest_paths_by_category = {
            label: sorted(path for path in node_by_path if path.startswith(f"{category_path}/"))
            for label, category_path in category_paths.items()
        }
        digest_paths = {path for paths in digest_paths_by_category.values() for path in paths}

        edge_keys: set[tuple[str, str, str | None]] = set()
        for source in digest_paths:
            for link in node_by_path[source].links:
                target = link.target_path
                if target in digest_paths or (target in node_by_path and target.startswith(daily_prefix)):
                    edge_keys.add((source, target, link.target_anchor))

        daily_paths = {target for _source, target, _anchor in edge_keys if target.startswith(daily_prefix)}

        nodes = [
            GraphSnapshotNode(
                id=f"virtual:{bucket.value}",
                path=category_paths[bucket],
                name=bucket.value,
                indexed=False,
                virtual=True,
            )
            for bucket in _CATEGORY_BUCKETS
        ]
        for bucket in _CATEGORY_BUCKETS:
            for path in digest_paths_by_category[bucket]:
                edge_keys.add((f"virtual:{bucket.value}", path, None))

        for path in sorted(digest_paths | daily_paths):
            node = node_by_path[path]
            nodes.append(
                GraphSnapshotNode(
                    id=path,
                    path=path,
                    name=node.front_matter.name,
                    description=node.front_matter.description,
                    indexed=True,
                ),
            )

        edges = [
            GraphSnapshotEdge(source=source, target=target, target_anchor=anchor)
            for source, target, anchor in sorted(edge_keys, key=lambda edge: (edge[0], edge[1], edge[2] or ""))
        ]
        graph = GraphSnapshot(nodes=nodes, edges=edges)

        self.context.response.success = True
        self.context.response.answer = graph.model_dump()
        self.logger.info(f"[{self.name}] nodes={len(nodes)} edges={len(edges)}")
        return self.context.response
