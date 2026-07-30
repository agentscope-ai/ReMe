"""Typed contracts for the daily-paper cookbook workflow."""

from typing import Literal

from pydantic import BaseModel, Field


class PaperInfo(BaseModel):
    """Normalized Hugging Face paper metadata plus local ranking fields."""

    arxiv_id: str
    title: str = ""
    summary: str = ""
    authors: list[str] = Field(default_factory=list)
    published_at: str | None = None
    submitted_on_daily_at: str | None = None
    upvotes: int = 0
    organization: str | None = None
    github_repo: str | None = None
    github_stars: int | None = None
    project_page: str | None = None
    thumbnail: str | None = None
    monthly_rank: int | None = None
    weekly_rank: int | None = None
    fused_score: float = 0.0

    @property
    def hf_url(self) -> str:
        """Return the canonical Hugging Face paper-page URL."""
        return f"https://huggingface.co/papers/{self.arxiv_id}"

    @property
    def arxiv_url(self) -> str:
        """Return the canonical arXiv abstract URL."""
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def pdf_url(self) -> str:
        """Return the canonical arXiv PDF download URL."""
        return f"https://arxiv.org/pdf/{self.arxiv_id}"


class SelectedPaper(BaseModel):
    """One agent-selected paper."""

    arxiv_id: str
    rank: int
    reason: str
    memory_relevance: Literal["high", "medium", "low"]


class PaperSelection(BaseModel):
    """Structured paper selection result."""

    selection_reasoning: str
    selected: list[SelectedPaper]
    alternates: list[str] = Field(default_factory=list)


class PaperNoteOutput(BaseModel):
    """Structured Claude Code output for one detailed paper note."""

    description: str
    body: str


class DailyBriefOutput(BaseModel):
    """Structured Claude Code output for the final five-minute brief."""

    description: str
    body: str
