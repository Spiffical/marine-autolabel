"""Differential test: ported parsers vs the originals in the legacy repo.

The port consolidated three duplicated click-validation loops and a repeated
last-tag JSON extraction, so "it looks equivalent" is not good enough. This runs
both implementations over an adversarial corpus and requires identical output.

Skipped unless MAL_LEGACY_REPO points at a checkout of the old repo, so CI and a
normal `pytest` run do not depend on it:

    MAL_LEGACY_REPO=~/path/to/sam3-autolabeling pytest tests/test_parity_parsers.py

The legacy module imports only stdlib at module scope, so it loads by path with
no sam3, torch or CUDA.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import random
import sys
from pathlib import Path

import pytest

from marine_autolabel.clickengine import parsing as new

LEGACY_ROOT = os.environ.get("MAL_LEGACY_REPO")
LEGACY_MODULE = (
    Path(LEGACY_ROOT).expanduser() / "nibi_model_compare" / "som_missed_creatures.py"
    if LEGACY_ROOT
    else None
)

pytestmark = pytest.mark.skipif(
    LEGACY_MODULE is None or not LEGACY_MODULE.is_file(),
    reason="set MAL_LEGACY_REPO to a checkout of the pre-migration repo",
)


@pytest.fixture(scope="module")
def legacy():
    spec = importlib.util.spec_from_file_location("legacy_som", LEGACY_MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["legacy_som"] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


def _answer(obj) -> str:
    return f"reasoning\n<answer>{json.dumps(obj)}</answer>"


def _click_corpus() -> list:
    coords = [0.5, 0.0, 1.0, -0.1, 1.1, True, False, None, "0.5", 3]
    labels = [0, 1, 2, -1, True, False]
    corpus = [
        {"x": x, "y": y, "label": lab}
        for x, y, lab in itertools.product(coords[:7], coords[:7], labels)
    ]
    corpus += [
        {"cell": [r, c], "label": 1}
        for r in (0, 5, 9, 10, -1, True, 1.7)
        for c in (0, 9, 10, -2)
    ]
    corpus += [
        {"cell": [3, 4]}, {"cell": "nope", "label": 1}, {"cell": [1], "label": 1},
        {"x": 0.5, "y": 0.5}, {}, None, "str", 42,
        {"x": True, "y": 0.2, "label": 1}, {"cell": [2, 2], "label": 0},
    ]
    return corpus


def _group_texts() -> list:
    random.seed(0)
    corpus = _click_corpus()
    groups = [
        {"id": gid, "description": desc, "clicks": random.sample(corpus, 5)}
        for gid in [1, 2, 0, -1, True, None, "1", 1.5, 99]
        for desc in ["a fish", None, 123, ""]
    ]
    groups += [
        {"id": 1, "clicks": []}, {"id": 1, "clicks": "nope"}, {"id": 1}, None, "x",
        {"id": 1, "clicks": [{"x": 0.5, "y": 0.5, "label": 0}]},
        {"id": 1, "clicks": [{"x": 0.5, "y": 0.5, "label": 0},
                             {"x": 0.6, "y": 0.6, "label": 1}]},
    ]
    texts = [_answer({"missed_creatures": groups[i:i + 5]}) for i in range(0, len(groups), 5)]
    texts += [
        _answer({"missed_creatures": groups}),
        _answer({"missed_creatures": "nope"}),
        _answer({"other": 1}),
        _answer([1, 2]),
        "no tag", "", None, 42, "<answer>bad json</answer>",
        _answer({"missed_creatures": []}),
        _answer({"missed_creatures": [{"id": 1, "clicks": [{"x": .1, "y": .1, "label": 1}]}] * 3}),
    ]
    return texts


@pytest.mark.parametrize("text", _group_texts(), ids=lambda t: str(t)[:28])
def test_creature_click_groups_match(legacy, text):
    assert legacy.parse_creature_click_groups(text) == new.parse_creature_click_groups(text)


MARK_TEXTS = [
    _answer({"accepted_marks": m})
    for m in ([1, 2, 3], [], [True, 1, "2", 2.0, None, 4], "nope", [0, -1, 99])
] + ["no tag", "", None, "<answer>bad</answer>"]


@pytest.mark.parametrize("text", MARK_TEXTS, ids=lambda t: str(t)[:28])
@pytest.mark.parametrize("valid_ids", [None, {1, 2}], ids=["all", "filtered"])
def test_som_response_matches(legacy, text, valid_ids):
    assert legacy.parse_som_response(text, valid_ids=valid_ids) == new.parse_som_response(
        text, valid_ids=valid_ids
    )


REFINE_TEXTS = [
    f"<refine>{json.dumps(p)}</refine>"
    for p in (
        {"action": "accept"}, {"action": "reject"},
        {"action": "refine", "add_points": []}, {"action": "refine"},
        {"action": "refine", "add_points": _click_corpus()[:12]},
        {"action": "bogus"}, {"action": "refine", "add_points": "nope"}, [1], "s",
    )
] + ["no tag", "", None, "<refine>bad</refine>"]


@pytest.mark.parametrize("text", REFINE_TEXTS, ids=lambda t: str(t)[:28])
def test_refinement_matches(legacy, text):
    assert legacy.parse_refinement_response(text) == new.parse_refinement_response(text)


CLICK_REFINE_TEXTS = [
    f"<click_refine>{json.dumps(p)}</click_refine>"
    for p in (
        {"action": "ok"}, {"action": "move", "new_clicks": _click_corpus()[:12]},
        {"action": "drop"}, {"action": "nope"}, {"action": "move"},
        {"action": "move", "new_clicks": "x"},
    )
] + ["no tag", "", None, "<click_refine>bad</click_refine>"]


@pytest.mark.parametrize("text", CLICK_REFINE_TEXTS, ids=lambda t: str(t)[:28])
def test_click_refinement_matches(legacy, text):
    assert legacy.parse_click_refinement_response(text) == new.parse_click_refinement_response(text)


VALIDITY_TEXTS = [
    "<validity>usable</validity>", "<validity>CORRUPTED</validity>", "nope", "",
    "<validity>usable</validity><validity>corrupted</validity>", None,
]


@pytest.mark.parametrize("text", VALIDITY_TEXTS, ids=lambda t: str(t)[:28])
def test_frame_validity_matches(legacy, text):
    assert legacy.parse_frame_validity(text) == new.parse_frame_validity(text)


# --------------------------------------------------------------------------
# The discovery prompt is tuned text. Generating its ordinals from the view
# list must not have changed a single character.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def legacy_prompt_fn():
    """Extract _mask_guided_discovery_prompt and exec it standalone."""
    import ast

    source = (Path(LEGACY_ROOT).expanduser() / "scripts" / "run_presentation_custom_flow.py")
    tree = ast.parse(source.read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_mask_guided_discovery_prompt"
    )
    namespace: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<legacy>", "exec"), namespace)
    return namespace["_mask_guided_discovery_prompt"]


@pytest.mark.parametrize("has_focus_crops", [False, True])
@pytest.mark.parametrize(
    "instruction", ["", "Sweep the left half. ", "Review the entire residual frame. "]
)
def test_discovery_prompt_is_byte_identical(legacy_prompt_fn, has_focus_crops, instruction):
    from marine_autolabel.clickengine.discovery import build_discovery_prompt

    assert legacy_prompt_fn(
        pass_instruction=instruction, has_focus_crops=has_focus_crops
    ) == build_discovery_prompt(
        pass_instruction=instruction, has_focus_crops=has_focus_crops
    )
