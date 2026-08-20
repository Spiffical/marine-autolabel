"""Tool-use telemetry from an agent run.

Ported from `scripts/analyze_agent_run.py`.

Answers a diagnostic question the mask counts cannot: did the agent loop
actually exercise its tools, or did it segment once and stop? A run can produce
plausible masks while never calling the verification tools at all, and that
looks identical in the output.

The debug history is written with `json.dumps(msg, indent=4)` per message, so it
is NOT valid JSONL -- pretty-printed objects span many lines. It is parsed by
scanning for consecutive top-level JSON objects instead.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

TOOL_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)

COUNTED_TOOLS = (
    "segment_phrase",
    "examine_each_mask",
    "drop_masks",
    "select_masks_and_return",
    "report_no_mask",
)


def iter_json_objects(text: str) -> list[dict[str, Any]]:
    """Every top-level JSON object in a concatenated stream.

    Tolerates the separators and stray text between objects that a
    pretty-printed debug log accumulates.
    """
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        try:
            obj, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            position += 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        position = end
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
    return objects


def parse_tool_calls(text: str) -> dict[str, Any]:
    """Summarise the tool calls in one agent history.

    Order is preserved: a run that calls `segment_phrase` five times before ever
    verifying tells a different story from one that alternates.
    """
    tools_in_order: list[str] = []
    segment_phrases: list[str] = []
    counts = dict.fromkeys(COUNTED_TOOLS, 0)

    for message in iter_json_objects(text):
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or []
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            for match in TOOL_RE.finditer(item.get("text", "")):
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                name = parsed.get("name")
                if not isinstance(name, str):
                    continue
                tools_in_order.append(name)
                if name in counts:
                    counts[name] += 1
                if name == "segment_phrase":
                    phrase = (parsed.get("parameters") or {}).get("text_prompt")
                    if isinstance(phrase, str):
                        segment_phrases.append(phrase)

    return {
        "tools_in_order": tools_in_order,
        "segment_phrases": segment_phrases,
        "n_tool_calls": len(tools_in_order),
        **{f"n_{name}": value for name, value in counts.items()},
    }


def parse_history_file(path: Path) -> dict[str, Any]:
    """Telemetry for one frame; a missing history is empty, not an error."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return parse_tool_calls("")
    return parse_tool_calls(text)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile. `q` in [0, 1]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def describe(values: list[float]) -> dict[str, float]:
    """Mean, median, p90, min and max. Empty gives zeros, never nan."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.9),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_run(per_frame: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-frame telemetry across a run.

    `never_verified` is the number worth watching: frames where the agent
    produced masks without ever calling a verification tool. Those masks are
    unexamined, and nothing downstream distinguishes them.
    """
    totals = {f"n_{name}": sum(f.get(f"n_{name}", 0) for f in per_frame) for name in COUNTED_TOOLS}
    never_verified = sum(
        1
        for f in per_frame
        if f.get("n_examine_each_mask", 0) == 0 and f.get("n_segment_phrase", 0) > 0
    )
    phrase_counts = [float(f.get("n_segment_phrase", 0)) for f in per_frame]
    return {
        "n_frames": len(per_frame),
        **totals,
        "never_verified_frames": never_verified,
        "segment_phrase_per_frame": describe(phrase_counts),
        "distinct_phrases": sorted({p for f in per_frame for p in f.get("segment_phrases", [])}),
    }
