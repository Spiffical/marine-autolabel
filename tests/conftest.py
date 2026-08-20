from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def run_summary() -> dict:
    return json.loads((FIXTURES / "run" / "summary.json").read_text())


@pytest.fixture(scope="session")
def firstpass_outputs() -> dict[str, dict]:
    return {
        p.name.replace("_frame_outputs_rle.json", ""): json.loads(p.read_text())
        for p in (FIXTURES / "firstpass").glob("*_frame_outputs_rle.json")
    }
