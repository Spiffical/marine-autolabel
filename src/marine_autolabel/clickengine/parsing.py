"""Parse the MLLM's tagged responses.

Ported from `nibi_model_compare/som_missed_creatures.py`. Every parser is
lenient by design: a malformed or missing tag yields "nothing", never an
exception. Callers routinely pass a response that may be None because the API
call failed, and a frame that parses to zero creatures is a normal outcome, not
an error.

Three parsers previously repeated the same click-validation loop and the same
last-tag-wins JSON extraction; both are factored out here. Behaviour is
unchanged, including the details that matter:

  * bool is a subclass of int in Python, so `True` must be rejected explicitly
    as a coordinate, a label and an id
  * when a tag appears more than once the LAST one wins -- it is the model's
    final answer after any reasoning
  * a group with no positive click is dropped: negative clicks constrain a
    target, they cannot seed one
"""
from __future__ import annotations

import json
import re
from typing import Any

GRID_N = 10
"""Cell coordinates are expressed against a 10x10 overlay grid."""

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_REFINE_RE = re.compile(r"<refine>\s*(.*?)\s*</refine>", re.DOTALL)
_CLICK_REFINE_RE = re.compile(r"<click_refine>\s*(.*?)\s*</click_refine>", re.DOTALL)
_VALIDITY_RE = re.compile(r"<validity>\s*(usable|corrupted)\s*</validity>", re.IGNORECASE)


def _is_number(value: Any) -> bool:
    """Numeric and not a bool -- `True` would otherwise pass as `1`."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _last_payload(text: str, pattern: re.Pattern[str]) -> Any:
    """JSON payload of the last matching tag, or None if absent/malformed."""
    if not isinstance(text, str) or not text:
        return None
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        return json.loads(matches[-1].strip())
    except json.JSONDecodeError:
        return None


def parse_click(entry: Any) -> dict[str, Any] | None:
    """Validate one click. Returns the normalised click, or None to drop it.

    Accepts normalised `{"x", "y"}` or a `{"cell": [row, col]}` alternative,
    which is converted to the centre of that grid cell. Labels must be 0 or 1.
    """
    if not isinstance(entry, dict):
        return None

    x = entry.get("x") if _is_number(entry.get("x")) else None
    y = entry.get("y") if _is_number(entry.get("y")) else None

    cell = entry.get("cell")
    if (x is None or y is None) and isinstance(cell, list) and len(cell) == 2:
        row, col = cell
        if _is_number(row) and _is_number(col):
            row, col = int(row), int(col)
            if 0 <= row < GRID_N and 0 <= col < GRID_N:
                x = (col + 0.5) / GRID_N
                y = (row + 0.5) / GRID_N

    if x is None or y is None:
        return None
    if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
        return None
    if entry.get("label") not in (0, 1):
        return None
    return {"x": float(x), "y": float(y), "label": int(entry["label"])}


def _parse_clicks(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [click for click in (parse_click(item) for item in raw) if click is not None]


def parse_som_response(text: str, *, valid_ids: set[int] | None = None) -> list[int]:
    """Accepted mark ids from `<answer>{"accepted_marks": [...]}</answer>`."""
    payload = _last_payload(text, _ANSWER_RE)
    raw = payload.get("accepted_marks") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    return [
        item
        for item in raw
        if isinstance(item, int)
        and not isinstance(item, bool)
        and (valid_ids is None or item in valid_ids)
    ]


def parse_click_proposals(text: str) -> list[dict[str, Any]]:
    """Legacy one-click-per-creature contract.

    Superseded by `parse_creature_click_groups`; kept for old run artefacts.
    """
    payload = _last_payload(text, _ANSWER_RE)
    raw = payload.get("missed_creatures") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []

    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        x, y = item.get("x"), item.get("y")
        if not _is_number(x) or not _is_number(y):
            continue
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            continue
        description = item.get("description")
        out.append(
            {
                "x": float(x),
                "y": float(y),
                "description": description if isinstance(description, str) else "",
            }
        )
    return out


def parse_creature_click_groups(text: str) -> list[dict[str, Any]]:
    """The grouped-click contract: one creature, many clicks.

    Groups losing all their clicks to validation are dropped, as are groups left
    with only negative clicks. Ids that are missing, non-positive or duplicated
    are reassigned to the lowest free integers, preserving valid ids in place.
    """
    payload = _last_payload(text, _ANSWER_RE)
    raw = payload.get("missed_creatures") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []

    groups: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        if not isinstance(item.get("clicks"), list):
            continue

        clicks = _parse_clicks(item["clicks"])
        if not any(click["label"] == 1 for click in clicks):
            continue

        raw_id = item.get("id")
        creature_id = (
            raw_id
            if isinstance(raw_id, int)
            and not isinstance(raw_id, bool)
            and raw_id > 0
            and raw_id not in seen_ids
            else None
        )
        seen_ids.add(creature_id)

        description = item.get("description")
        groups.append(
            {
                "id": creature_id,
                "description": description if isinstance(description, str) else "",
                "clicks": clicks,
            }
        )

    if any(group["id"] is None for group in groups):
        assigned = {group["id"] for group in groups if group["id"] is not None}
        counter = 1
        for group in groups:
            if group["id"] is None:
                while counter in assigned:
                    counter += 1
                group["id"] = counter
                assigned.add(counter)
                counter += 1
    return groups


def parse_frame_validity(text: str) -> str | None:
    """`'usable'`, `'corrupted'`, or None when the tag is missing."""
    matches = _VALIDITY_RE.findall(text or "")
    return matches[-1].lower() if matches else None


def parse_refinement_response(text: str) -> dict[str, Any] | None:
    """`<refine>{"action": "accept"|"reject"|"refine", "add_points": [...]}</refine>`.

    A "refine" action without an `add_points` list is malformed and yields None
    -- it asks for refinement while supplying nothing to refine with.
    """
    payload = _last_payload(text, _REFINE_RE)
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    if action not in ("accept", "reject", "refine"):
        return None
    if action != "refine":
        return {"action": action, "add_points": []}
    if not isinstance(payload.get("add_points"), list):
        return None
    return {"action": "refine", "add_points": _parse_clicks(payload["add_points"])}


def parse_click_refinement_response(text: str) -> dict[str, Any] | None:
    """`<click_refine>{"action": "ok"|"move"|"drop", "new_clicks": [...]}</click_refine>`.

    "ok" keeps the group, "move" replaces its clicks, "drop" discards it.
    """
    payload = _last_payload(text, _CLICK_REFINE_RE)
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    if action not in ("ok", "move", "drop"):
        return None
    return {"action": action, "new_clicks": _parse_clicks(payload.get("new_clicks") or [])}
