"""Procedural memory workflow."""

from ...core import R

from .dump_memory import DumpMemory
from .load_memory import LoadMemory

from .summary.trajectory_preprocess import TrajectoryPreprocess
from .summary.trajectory_segmentation import TrajectorySegmentation
from .summary.success_extraction import SuccessExtraction
from .summary.failure_extraction import FailureExtraction
from .summary.comparative_extraction import ComparativeExtraction
from .summary.memory_validation import MemoryValidation
from .summary.memory_deduplication import MemoryDeduplication
from .summary.memory_addition import MemoryAddition

from .retrieve.build_query import BuildQuery
from .retrieve.memory_deletion import MemoryDeletion
from .retrieve.memory_retrieval import MemoryRetrieval
from .retrieve.merge_memory import MergeMemory
from .retrieve.rerank_memory import RerankMemory
from .retrieve.rewrite_memory import RewriteMemory
from .retrieve.update_memory_metadata import UpdateMemoryMetadata

__all__ = [
    "DumpMemory",
    "LoadMemory",
    "TrajectoryPreprocess",
    "TrajectorySegmentation",
    "SuccessExtraction",
    "FailureExtraction",
    "ComparativeExtraction",
    "MemoryValidation",
    "MemoryDeduplication",
    "MemoryAddition",
    "BuildQuery",
    "MemoryDeletion",
    "MemoryRetrieval",
    "MergeMemory",
    "RerankMemory",
    "RewriteMemory",
    "UpdateMemoryMetadata",
]

for name in __all__:
    tool_class = globals()[name]
    R.ops.register()(tool_class)
