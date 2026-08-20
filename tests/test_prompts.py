"""Prompt loading and the general/underwater profile invariant.

A missing profile variant would only surface partway through an expensive run,
so the pairing is enforced here instead.
"""
from __future__ import annotations

from importlib import resources

import pytest

from marine_autolabel import prompts

ALL = sorted(p.name[:-4] for p in resources.files(prompts).iterdir() if p.name.endswith(".txt"))
PAIRED = sorted({n.rsplit("_", 1)[0] for n in ALL if n.endswith(("_general", "_underwater"))})
STANDALONE = sorted(n for n in ALL if not n.endswith(("_general", "_underwater")))


def test_prompts_were_actually_shipped_as_package_data():
    assert len(ALL) >= 30, f"expected the full prompt set, found {len(ALL)}"


@pytest.mark.parametrize("name", PAIRED)
@pytest.mark.parametrize("profile", ["general", "underwater"])
def test_every_paired_prompt_resolves_in_both_profiles(name, profile):
    text = prompts.load(name, profile=profile)
    assert text.strip(), f"{name} ({profile}) is empty"


@pytest.mark.parametrize("name", PAIRED)
def test_the_two_profiles_actually_differ(name):
    """If they were identical the split would be dead weight."""
    assert prompts.load(name, "general") != prompts.load(name, "underwater")


@pytest.mark.parametrize("name", STANDALONE)
def test_standalone_prompts_fall_back_across_profiles(name):
    assert prompts.load(name, "underwater") == prompts.load(name, "general")


def test_the_settled_stages_all_have_a_prompt():
    for name in (
        "som_click_discovery",
        "som_click_refinement",
        "som_frame_quality",
        "som_missed_creature",
        "missed_creature_verify",
        "temporal_discovery",
    ):
        assert prompts.load(name, "underwater").strip()


def test_rejects_unknown_profile():
    with pytest.raises(ValueError, match="unknown profile"):
        prompts.load("som_click_discovery", profile="martian")


def test_missing_prompt_is_an_error():
    with pytest.raises(FileNotFoundError):
        prompts.load("does_not_exist", profile="underwater")
