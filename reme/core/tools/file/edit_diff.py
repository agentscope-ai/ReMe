"""Diff utilities for edit tool."""

import re
from dataclasses import dataclass
from difflib import unified_diff


def detect_line_ending(content: str) -> str:
    """Detect line ending style (CRLF or LF)."""
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    """Normalize line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    """Restore original line endings."""
    return text.replace("\n", ending) if ending == "\r\n" else text


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching: strip trailing whitespace, normalize quotes/dashes."""
    lines = text.split("\n")
    normalized = "\n".join(line.rstrip() for line in lines)

    # Smart quotes → ASCII
    normalized = re.sub(r"[\u2018\u2019\u201A\u201B]", "'", normalized)
    normalized = re.sub(r"[\u201C\u201D\u201E\u201F]", '"', normalized)

    # Dashes → hyphen
    normalized = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", normalized)

    # Special spaces → regular space
    normalized = re.sub(r"[\u00A0\u2002-\u200A\u202F\u205F\u3000]", " ", normalized)

    return normalized


@dataclass
class FuzzyMatchResult:
    """Result of fuzzy text matching."""

    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """Find old_text in content, trying exact match first, then fuzzy match."""
    # Try exact match
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    # Try fuzzy match
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)

    if fuzzy_index == -1:
        return FuzzyMatchResult(
            found=False,
            index=-1,
            match_length=0,
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    return FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old_text),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


def strip_bom(content: str) -> tuple[str, str]:
    """Strip UTF-8 BOM, return (bom, text_without_bom)."""
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


@dataclass
class DiffResult:
    """Result of diff generation."""

    diff: str
    first_changed_line: int | None


def generate_diff_string(old_content: str, new_content: str, context_lines: int = 4) -> DiffResult:
    """Generate unified diff with line numbers."""
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")

    # Use difflib to get the changes
    diff_lines = list(
        unified_diff(
            old_lines,
            new_lines,
            lineterm="",
            n=context_lines,
        ),
    )

    if not diff_lines:
        return DiffResult(diff="", first_changed_line=None)

    # Parse and format the diff
    output = []
    first_changed_line = None
    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))

    old_line_num = 1
    new_line_num = 1

    for line in diff_lines[2:]:  # Skip header lines
        if line.startswith("@@"):
            # Parse hunk header
            match = re.match(r"@@ -(\d+),?\d* \+(\d+),?\d* @@", line)
            if match:
                old_line_num = int(match.group(1))
                new_line_num = int(match.group(2))
            continue

        if line.startswith("+"):
            if first_changed_line is None:
                first_changed_line = new_line_num
            line_num = str(new_line_num).rjust(line_num_width)
            output.append(f"+{line_num} {line[1:]}")
            new_line_num += 1
        elif line.startswith("-"):
            if first_changed_line is None:
                first_changed_line = new_line_num
            line_num = str(old_line_num).rjust(line_num_width)
            output.append(f"-{line_num} {line[1:]}")
            old_line_num += 1
        else:
            # Context line
            line_num = str(old_line_num).rjust(line_num_width)
            output.append(f" {line_num} {line[1:] if line.startswith(' ') else line}")
            old_line_num += 1
            new_line_num += 1

    return DiffResult(diff="\n".join(output), first_changed_line=first_changed_line)
