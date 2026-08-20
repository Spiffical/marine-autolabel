"""Loading a frame's inputs: first-pass masks and prior accepted masks.

Ported from `run_presentation_custom_flow.py`.

The checks here are deliberately strict and fail loudly. A first pass run with
the wrong model, or with recorded errors, silently poisons every downstream
number, and these runs are expensive enough that discovering it afterwards is
worse than refusing to start.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..rle import decode_rle_to_mask

SOURCE_FIRSTPASS = "firstpass"
SOURCE_INITIAL = "initial"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_firstpass(
    frame_dir: Path, expected_model: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Masks from a first-pass run, with its summary.

    An empty first pass is a legitimate result, not an error. On a dense scene
    the text stage can ground nothing at all, leaving the click engine to
    recover every organism.
    """
    summary = read_json(frame_dir / "summary.json")
    actual_model = str(summary.get("model", ""))
    if actual_model != expected_model:
        raise RuntimeError(
            f"first-pass model mismatch in {frame_dir}: "
            f"expected {expected_model!r}, found {actual_model!r}"
        )
    if int(summary.get("error_count", 0)) != 0:
        raise RuntimeError(f"first-pass run has errors: {frame_dir}")

    doc = read_json(frame_dir / "frame_outputs_rle.json")
    frames = doc.get("frames", [])
    if len(frames) != 1:
        raise RuntimeError(
            f"expected exactly one first-pass frame in {frame_dir}, found {len(frames)}"
        )

    height, width = (int(v) for v in doc["frame_size_hw"])
    record = frames[0]
    probs = list(record.get("out_probs") or [])
    boxes = list(record.get("out_boxes_xywh") or [])

    masks = []
    for index, rle in enumerate(record.get("out_binary_masks_rle") or []):
        masks.append(
            {
                "mask": decode_rle_to_mask(rle, height, width).astype(bool),
                "prob": float(probs[index]) if index < len(probs) else None,
                "box_xywh": boxes[index] if index < len(boxes) else None,
                "source": SOURCE_FIRSTPASS,
            }
        )
    return masks, summary


def load_initial_masks(
    root: Path | None, frame_id: str, expected_shape: tuple[int, int]
) -> list[dict[str, Any]]:
    """Accepted masks from a prior run, so residual discovery can continue.

    Prefers the final mask set, falling back to a checkpoint when a run was
    interrupted before it finished.
    """
    if root is None:
        return []

    path = root / frame_id / "final_masks_rle.json"
    if not path.exists():
        path = root / frame_id / "checkpoint_masks_rle.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no final_masks_rle.json or checkpoint_masks_rle.json for {frame_id} under {root}"
        )

    doc = read_json(path)
    height, width = (int(value) for value in doc["frame_size_hw"])
    if (height, width) != expected_shape:
        raise RuntimeError(
            f"initial mask/frame size mismatch for {frame_id}: "
            f"{(height, width)} vs {expected_shape}"
        )
    return [
        {"mask": decode_rle_to_mask(rle, height, width).astype(bool), "source": SOURCE_INITIAL}
        for rle in doc.get("masks", [])
    ]
