"""Aggregating repeated runs.

Ported from `scripts/summarize_custom_flow_repeats.py`.

The methodology rule this enforces: **evaluate any MLLM-in-the-loop change at
repeats >= 3 and report mean +/- std.** A single run is a plumbing check, not a
result. `summarize` refuses fewer than three repeats rather than producing a
number that looks comparable but is not.

The original module's title is worth keeping in view: "without treating mask
count as recall". More masks is not better if the extra ones are wrong, and
ground truth here is incomplete, so these are descriptive statistics about run
behaviour, not a score.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

MIN_REPEATS = 3

DEFAULT_METRICS = (
    "n_firstpass",
    "n_text_verified",
    "n_recovered",
    "n_final",
    "runtime_sec",
)

COMPARABILITY_KEYS = (
    "model",
    "firstpass_model",
    "text_proposal_mode",
    "finder_mode",
    "mask_generator",
    "strategy",
)


def describe(values: list[float]) -> dict[str, float]:
    """Mean, standard deviation, min and max.

    Standard deviation of a single value is 0.0 rather than an error, but a
    single value should not reach here -- `summarize` enforces the repeat count.
    """
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def format_mean_std(stats: dict[str, float], places: int = 2) -> str:
    """`mean +/- std`, the reporting form the methodology asks for."""
    return f"{stats['mean']:.{places}f} +/- {stats['std']:.{places}f}"


def load_repeats(root: Path, pattern: str = "repeat_*/summary.json") -> list[dict[str, Any]]:
    """Read every repeat summary under `root`, in sorted order."""
    paths = sorted(Path(root).glob(pattern))
    if len(paths) < MIN_REPEATS:
        raise ValueError(
            f"expected at least {MIN_REPEATS} repeats under {root}; found {len(paths)}. "
            "A single run is a plumbing check, not a result."
        )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def check_comparable(docs: list[dict[str, Any]]) -> None:
    """Refuse to average runs that were not the same experiment.

    Two things must match: the ordered frame list, and the configuration keys
    that change what the pipeline does. Averaging across a changed model or
    generator silently produces a number describing neither.
    """
    orders = [[str(row["frame_id"]) for row in doc.get("frames", [])] for doc in docs]
    if any(order != orders[0] for order in orders[1:]):
        raise ValueError("repeat summaries do not contain the same ordered frame IDs")

    for key in COMPARABILITY_KEYS:
        values = {json.dumps(doc.get("config", doc).get(key), sort_keys=True) for doc in docs}
        if len(values) > 1:
            raise ValueError(
                f"repeats disagree on {key!r}: {sorted(values)}. These are different "
                "experiments and must not be averaged."
            )


def summarize(
    root: Path, metrics: tuple[str, ...] = DEFAULT_METRICS
) -> dict[str, Any]:
    """Aggregate repeats into per-metric mean/std/min/max."""
    docs = load_repeats(root)
    check_comparable(docs)

    aggregate: dict[str, Any] = {"n_repeats": len(docs), "metrics": {}, "per_frame": {}}
    for metric in metrics:
        values = [float(doc[metric]) for doc in docs if isinstance(doc.get(metric), (int, float))]
        if values:
            aggregate["metrics"][metric] = describe(values)

    for index, frame_id in enumerate(str(row["frame_id"]) for row in docs[0].get("frames", [])):
        per_frame: dict[str, Any] = {}
        for metric in metrics:
            values = [
                float(doc["frames"][index][metric])
                for doc in docs
                if isinstance(doc["frames"][index].get(metric), (int, float))
            ]
            if values:
                per_frame[metric] = describe(values)
        aggregate["per_frame"][frame_id] = per_frame
    return aggregate
