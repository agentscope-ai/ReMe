"""Daily Paper plugin for ReMe."""

from .analyze import DailyPaperAnalyzeStep
from .collect import DailyPaperCollectStep
from .digest import DailyPaperDigestStep
from .rank import DailyPaperRankStep
from .schema import AnalyzedPaper, DailyPaperMarkdownOutput, PaperInfo, PaperPick, PaperPickList
from .select import DailyPaperSelectStep

__all__ = [
    "DailyPaperAnalyzeStep",
    "DailyPaperCollectStep",
    "DailyPaperDigestStep",
    "DailyPaperRankStep",
    "DailyPaperSelectStep",
    "AnalyzedPaper",
    "DailyPaperMarkdownOutput",
    "PaperInfo",
    "PaperPick",
    "PaperPickList",
]
