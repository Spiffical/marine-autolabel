"""Reading and merging the `frame_outputs_rle` run format.

Ported from `nibi_model_compare/frame_output_utils.py`. The optional-import
guards are gone (numpy, opencv and torch handling are settled by packaging);
everything else is unchanged.

The format is what the first-pass agent writes and what every later stage reads,
so it is the pipeline's real data contract. Masks may arrive as `out_binary_masks`
with parallel `out_obj_ids`, or under any of several legacy keys; `iter_output_masks_with_ids`
normalises all of them.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:  # torch is only present with the [sam3] extra; outputs may be tensors.
    import torch
except ImportError:  # pragma: no cover - exercised only without the extra
    torch = None


def _is_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def read_video_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if _is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except Exception:
        return []


def _mask_list_from_outputs(outputs: dict[str, Any]) -> list[np.ndarray]:
    if not isinstance(outputs, dict):
        return []
    masks = outputs.get("pred_masks")
    if masks is None:
        masks = outputs.get("video_res_masks")
    if masks is None:
        masks = outputs.get("out_binary_masks")
    if masks is None and isinstance(outputs.get("obj_id_to_mask"), dict):
        masks = list(outputs["obj_id_to_mask"].values())
    if masks is None:
        return []
    if isinstance(masks, dict):
        masks = list(masks.values())
    if _is_tensor(masks):
        masks = masks.detach().cpu().numpy()

    parsed: list[np.ndarray] = []
    for mask in masks:
        arr = np.asarray(mask)
        while arr.ndim > 2:
            arr = arr[0]
        parsed.append(arr > 0)
    return parsed


def iter_output_masks_with_ids(
    outputs: dict[str, Any], frame_h: int, frame_w: int
) -> list[tuple[int, np.ndarray]]:
    if not isinstance(outputs, dict):
        return []

    if "out_binary_masks" in outputs:
        raw_masks = outputs.get("out_binary_masks")
        raw_ids = outputs.get("out_obj_ids")
        if _is_tensor(raw_masks):
            raw_masks = raw_masks.detach().cpu().numpy()
        if _is_tensor(raw_ids):
            raw_ids = raw_ids.detach().cpu().numpy()
        if raw_masks is None:
            return []

        out: list[tuple[int, np.ndarray]] = []
        for i, raw_mask in enumerate(raw_masks):
            arr = np.asarray(raw_mask)
            while arr.ndim > 2:
                arr = arr[0]
            if arr.shape != (frame_h, frame_w):
                arr = cv2.resize(
                    arr.astype(np.float32),
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = arr > 0.5
            if raw_ids is not None and len(raw_ids) > i:
                obj_id = int(raw_ids[i])
            else:
                obj_id = i + 1
            out.append((obj_id, mask))
        return out

    return [(i + 1, m) for i, m in enumerate(_mask_list_from_outputs(outputs))]


def frame_object_metadata(
    outputs: dict[str, Any], frame_h: int, frame_w: int
) -> dict[int, dict[str, Any]]:
    obj_ids = _to_list(outputs.get("out_obj_ids"))
    out_boxes_xywh = _to_list(outputs.get("out_boxes_xywh"))
    out_probs = _to_list(outputs.get("out_probs"))
    out_tracker_probs = _to_list(outputs.get("out_tracker_probs"))

    metadata: dict[int, dict[str, Any]] = {}
    for i, raw_obj_id in enumerate(obj_ids):
        try:
            obj_id = int(raw_obj_id)
        except Exception:
            continue

        box_xywh: list[float] | None = None
        box_xyxy: tuple[int, int, int, int] | None = None
        if i < len(out_boxes_xywh):
            raw_box = out_boxes_xywh[i]
            if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
                try:
                    x, y, w, h = [float(raw_box[j]) for j in range(4)]
                    box_xywh = [x, y, w, h]
                    x1 = int(round(x * frame_w))
                    y1 = int(round(y * frame_h))
                    x2 = int(round((x + max(0.0, w)) * frame_w))
                    y2 = int(round((y + max(0.0, h)) * frame_h))
                    x1 = max(0, min(frame_w - 1, x1))
                    y1 = max(0, min(frame_h - 1, y1))
                    x2 = max(0, min(frame_w - 1, x2))
                    y2 = max(0, min(frame_h - 1, y2))
                    if x2 > x1 and y2 > y1:
                        box_xyxy = (x1, y1, x2, y2)
                except Exception:
                    box_xywh = None
                    box_xyxy = None

        confidence: float | None = None
        if i < len(out_probs):
            raw_prob = out_probs[i]
            if isinstance(raw_prob, (list, tuple)) and len(raw_prob) > 0:
                raw_prob = raw_prob[0]
            try:
                confidence = float(raw_prob)
            except Exception:
                confidence = None

        tracker_confidence: float | None = None
        if i < len(out_tracker_probs):
            raw_tracker_prob = out_tracker_probs[i]
            if isinstance(raw_tracker_prob, (list, tuple)) and len(raw_tracker_prob) > 0:
                raw_tracker_prob = raw_tracker_prob[0]
            try:
                tracker_confidence = float(raw_tracker_prob)
            except Exception:
                tracker_confidence = None

        metadata[obj_id] = {
            "box_xywh": box_xywh,
            "box_xyxy": box_xyxy,
            "confidence": confidence,
            "tracker_confidence": tracker_confidence,
        }

    return metadata


def merge_frame_outputs_by_obj_ids(
    base: dict[str, Any], patch: dict[str, Any], replace_obj_ids: set[int]
) -> dict[str, Any]:
    replace_obj_ids = {int(x) for x in replace_obj_ids}
    base_outputs = dict(base or {})
    patch_outputs = dict(patch or {})

    base_masks = _to_list(base_outputs.get("out_binary_masks"))
    patch_masks = _to_list(patch_outputs.get("out_binary_masks"))

    if base_masks:
        sample_mask = np.asarray(base_masks[0])
        while sample_mask.ndim > 2:
            sample_mask = sample_mask[0]
        frame_h, frame_w = int(sample_mask.shape[0]), int(sample_mask.shape[1])
    elif patch_masks:
        sample_mask = np.asarray(patch_masks[0])
        while sample_mask.ndim > 2:
            sample_mask = sample_mask[0]
        frame_h, frame_w = int(sample_mask.shape[0]), int(sample_mask.shape[1])
    else:
        frame_h = frame_w = 0

    base_meta = frame_object_metadata(base_outputs, frame_h, frame_w)
    patch_meta = frame_object_metadata(patch_outputs, frame_h, frame_w)

    combined_masks: dict[int, np.ndarray] = {}
    for obj_id, mask in iter_output_masks_with_ids(base_outputs, frame_h, frame_w):
        if obj_id not in replace_obj_ids:
            combined_masks[int(obj_id)] = np.asarray(mask).astype(bool)

    for obj_id, mask in iter_output_masks_with_ids(patch_outputs, frame_h, frame_w):
        if obj_id in replace_obj_ids:
            combined_masks[int(obj_id)] = np.asarray(mask).astype(bool)

    final_obj_ids = sorted(combined_masks.keys())
    merged = dict(base_outputs)

    if final_obj_ids:
        merged["out_obj_ids"] = np.asarray(final_obj_ids, dtype=np.int64)
        merged["out_binary_masks"] = np.stack(
            [combined_masks[obj_id].astype(bool) for obj_id in final_obj_ids], axis=0
        )

        merged["out_probs"] = np.asarray(
            [
                float(
                    (patch_meta if obj_id in patch_meta else base_meta)
                    .get(obj_id, {})
                    .get("confidence", 0.0)
                    or 0.0
                )
                for obj_id in final_obj_ids
            ],
            dtype=np.float32,
        )

        if any(
            (patch_meta if obj_id in patch_meta else base_meta)
            .get(obj_id, {})
            .get("tracker_confidence")
            is not None
            for obj_id in final_obj_ids
        ):
            merged["out_tracker_probs"] = np.asarray(
                [
                    float(
                        (patch_meta if obj_id in patch_meta else base_meta)
                        .get(obj_id, {})
                        .get("tracker_confidence", 0.0)
                        or 0.0
                    )
                    for obj_id in final_obj_ids
                ],
                dtype=np.float32,
            )

        if any(
            (patch_meta if obj_id in patch_meta else base_meta)
            .get(obj_id, {})
            .get("box_xywh")
            is not None
            for obj_id in final_obj_ids
        ):
            merged["out_boxes_xywh"] = np.asarray(
                [
                    (
                        (patch_meta if obj_id in patch_meta else base_meta)
                        .get(obj_id, {})
                        .get("box_xywh")
                        or [0.0, 0.0, 0.0, 0.0]
                    )
                    for obj_id in final_obj_ids
                ],
                dtype=np.float32,
            )
    else:
        merged["out_obj_ids"] = np.zeros(0, dtype=np.int64)
        merged["out_binary_masks"] = np.zeros((0, frame_h, frame_w), dtype=bool)
        merged["out_probs"] = np.zeros(0, dtype=np.float32)
        if "out_tracker_probs" in merged or "out_tracker_probs" in patch_outputs:
            merged["out_tracker_probs"] = np.zeros(0, dtype=np.float32)
        if "out_boxes_xywh" in merged or "out_boxes_xywh" in patch_outputs:
            merged["out_boxes_xywh"] = np.zeros((0, 4), dtype=np.float32)

    if "frame_stats" in patch_outputs:
        merged["frame_stats"] = patch_outputs["frame_stats"]

    return merged


# --------------------------------------------------------------------------
# On-disk format
#
# `out_binary_masks_rle` is how frames are stored; `out_binary_masks` is the
# in-memory array form SAM3 produces at runtime. The helpers above take the
# latter. In the original repo the conversion between them was open-coded at
# every call site (8+ of them, in click_engine_probe, run_presentation_custom_flow,
# som_missed_creatures, analyze_agent_run, render_video_from_frame_outputs, ...).
# These two functions are that missing seam; they add no behaviour, they just
# put the decode in one place so later phases have something to call.
# --------------------------------------------------------------------------


def load_frame_outputs(path: str) -> dict[str, Any]:
    """Read a `frame_outputs_rle.json` file."""
    import json

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def decode_frame_row(
    row: dict[str, Any], frame_h: int, frame_w: int
) -> list[tuple[int, np.ndarray]]:
    """Decode one stored frame row into `(obj_id, mask)` pairs.

    Accepts either the stored RLE form (`out_binary_masks_rle`) or the in-memory
    array form (`out_binary_masks`), so callers do not have to care which they hold.
    """
    from .rle import decode_rle_to_mask

    rles = row.get("out_binary_masks_rle")
    if rles is None:
        return iter_output_masks_with_ids(row, frame_h, frame_w)

    obj_ids = _to_list(row.get("out_obj_ids"))
    out: list[tuple[int, np.ndarray]] = []
    for index, rle in enumerate(rles):
        mask = decode_rle_to_mask(rle, frame_h, frame_w).astype(bool)
        obj_id = int(obj_ids[index]) if index < len(obj_ids) else index + 1
        out.append((obj_id, mask))
    return out
