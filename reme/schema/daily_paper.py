"""Typed contracts for the daily-paper cookbook workflow."""

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


class PaperPick(BaseModel):
    """One minimal paper selection returned by the agent."""

    arxiv_id: str
    reasoning: str


class PaperPickList(BaseModel):
    """The ordered papers selected for detailed analysis."""

    papers: list[PaperPick]


class DailyPaperMarkdownOutput(BaseModel):
    """One Chinese Markdown document returned by a tool-free agent."""

    title: str
    desc: str
    body: str


class AnalyzedPaper(DailyPaperMarkdownOutput):
    """A persisted paper analysis passed directly to the digest step."""

    arxiv_id: str
    reasoning: str
    note_path: str
    pdf_path: str
