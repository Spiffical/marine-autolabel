"""System prompts, loaded as package data.

Every prompt exists as a `<name>_general.txt` / `<name>_underwater.txt` pair.
The profile is selected by config, not by filename convention at the call site --
this is the seam that lets the pipeline run on non-marine imagery.
"""
from __future__ import annotations

from importlib import resources

VALID_PROFILES = ("general", "underwater")


def load(name: str, profile: str = "underwater") -> str:
    """Return the text of prompt `name` for `profile`.

    Falls back to the bare `<name>.txt` when no profile variant exists.
    """
    if profile not in VALID_PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {VALID_PROFILES}")
    root = resources.files(__name__)
    for candidate in (f"{name}_{profile}.txt", f"{name}.txt"):
        path = root.joinpath(candidate)
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"no prompt {name!r} for profile {profile!r}")
