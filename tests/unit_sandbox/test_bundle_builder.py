"""Focused coverage for the sealed benchmark bundle."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = PROJECT_ROOT / "meta-reme/bundle_builder.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("reme_bundle_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_prunes_unused_runtime_backends(tmp_path: Path) -> None:
    """Generated benchmark bundles omit backends outside the sealed runtime contract."""
    builder = _load_builder()
    bundle = builder.build_bundle("default", tmp_path, validate=True)

    forbidden = (
        "reme/components/client/http_client.py",
        "reme/components/client/mcp_client.py",
        "reme/components/file_graph/neo4j_file_graph.py",
        "reme/components/file_graph/nx_file_graph.py",
        "reme/components/file_store/faiss_local_file_store.py",
        "reme/components/file_store/zvec_local_file_store.py",
        "reme/components/service/http_service.py",
        "reme/components/service/mcp_service.py",
        "reme/components/tokenizer/jieba_tokenizer.py",
        "reme/schema/auto_fin.py",
        "reme/schema/daily_paper.py",
        "reme/steps/common/add.py",
        "reme/steps/common/demo.py",
        "reme/steps/common/llm_demo.py",
        "reme/steps/common/stream_demo.py",
        "reme/steps/common/stream_llm_demo.py",
        "reme/steps/common/version.py",
        "reme/steps/file_io/read_image.py",
        "reme/utils/arxiv.py",
        "reme/utils/huggingface_papers.py",
    )
    assert all(not (bundle / path).exists() for path in forbidden)

    config = yaml.safe_load((bundle / "reme/config/default.yaml").read_text(encoding="utf-8"))
    assert config["service"]["backend"] == "cli"
    assert "read_image" not in config["jobs"]
    assert "version" not in config["jobs"]
    assert {"dream_cron", "optimize_index_cron"}.issubset(config["jobs"])
    assert not list(bundle.rglob("__pycache__"))
    assert not (bundle / "README.md").exists()
    assert not (bundle / "LICENSE").exists()
    assert (bundle / "pyproject.toml").is_file()

    embedding_source = (bundle / "reme/components/as_embedding/__init__.py").read_text(encoding="utf-8")
    assert "dashscope_multimodal" not in embedding_source

    assert (bundle / "reme/steps/evolve/auto_memory_cc.py").is_file()
    benchmark_bundle = builder.build_bundle("lme", tmp_path / "lme", validate=True)
    assert not (benchmark_bundle / "reme/steps/evolve/auto_memory_cc.py").exists()
    assert "AutoMemoryCCStep" not in (benchmark_bundle / "reme/steps/evolve/__init__.py").read_text(encoding="utf-8")
    assert not list(benchmark_bundle.rglob("__pycache__"))
