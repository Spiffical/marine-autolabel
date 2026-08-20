"""Shared detection of machine-specific detail in tracked files.

Matched by SHAPE rather than by listing real hostnames, usernames or addresses --
a guard that names what it forbids republishes it.
"""
from __future__ import annotations

import re

# An ssh alias on its own ("ssh gpu") is deliberately NOT matched: it is not
# sensitive by itself, and detecting it by regex fires on prose like
# "set MAL_REMOTE_HOST (ssh alias or user@host)". What matters is a real
# path, address or scp target, and those are matched below.
PATTERNS = {
    "absolute home path": re.compile(r"/(home|Users)/[A-Za-z0-9._-]+"),
    "host:port": re.compile(r"\b[A-Za-z0-9._-]+:(22|2200|8022)\b"),
    "private IPv4": re.compile(r"\b(10|100|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "scp-style target": re.compile(r"\b[a-z][a-z0-9_-]*:/(?:home|Users|opt|srv)/"),
}


def find(text: str) -> list[tuple[str, str]]:
    """Return (kind, match) for every machine-specific detail found."""
    hits = []
    for kind, pattern in PATTERNS.items():
        for match in pattern.findall(text):
            hits.append((kind, match if isinstance(match, str) else match[0]))
    return hits
