"""Documentation invariants -- cheap, and they have already caught leaks."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from hostcheck import find as find_host_detail

ROOT = Path(__file__).parent.parent
TRACKED_DOCS = [
    *sorted(ROOT.glob("docs/*.md")),
    ROOT / "README.md",
    ROOT / "AGENTS.md",
]


@pytest.mark.parametrize("path", TRACKED_DOCS, ids=lambda p: p.name)
def test_no_machine_specific_detail_in_tracked_docs(path):
    if path.name == "LOCAL_NOTES.md":
        pytest.skip("LOCAL_NOTES.md is untracked by design")
    hits = find_host_detail(path.read_text())
    assert not hits, f"{path.name} carries machine-specific detail: {hits}"


def test_local_notes_are_gitignored():
    assert "docs/LOCAL_NOTES.md" in (ROOT / ".gitignore").read_text()


@pytest.mark.parametrize("name", ["architecture.md", "running.md"])
def test_the_entry_documents_exist(name):
    assert (ROOT / "docs" / name).is_file()


def test_readme_points_at_them():
    readme = (ROOT / "README.md").read_text()
    for name in ("architecture.md", "running.md"):
        assert name in readme


RESULT_CLAIM = re.compile(
    r"\b(recall|precision|IoU|mAP)\b[^.\n]{0,24}\b0\.\d{2,}", re.IGNORECASE
)


def test_no_unpublished_results_are_quoted_in_tracked_docs():
    """Findings and presentation material are kept in private/, which is
    gitignored. This repository ships the pipeline, not the results.

    Matched by SHAPE -- a metric name near a decimal -- rather than by listing
    the actual figures, which would republish them in the guard itself.
    """
    for path in TRACKED_DOCS:
        hit = RESULT_CLAIM.search(path.read_text())
        assert hit is None, f"{path.name} quotes a result: {hit.group(0)!r}"


def test_private_material_is_gitignored():
    assert "private/" in (ROOT / ".gitignore").read_text()
