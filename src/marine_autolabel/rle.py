"""COCO RLE encode/decode for binary masks.

Ported from `nibi_model_compare/frame_output_utils.py`. Behaviour is unchanged;
the optional-import guards are gone because numpy and pycocotools are hard
dependencies of this package.

The wire format used throughout the pipeline stores `counts` as a str (JSON does
not carry bytes), while pycocotools works in bytes. Both directions are handled
here so no caller has to think about it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools import mask as mask_util


def decode_rle_to_mask(rle: Any, height: int, width: int) -> np.ndarray:
    """Decode a COCO RLE (str or dict, bytes or str counts) to a 2-D array."""
    if isinstance(rle, str):
        rle = {"counts": rle.encode("utf-8"), "size": [height, width]}
    elif isinstance(rle, dict) and "counts" in rle and isinstance(rle["counts"], str):
        rle = dict(rle)
        rle["counts"] = rle["counts"].encode("utf-8")

    decoded = mask_util.decode([rle])
    if decoded.ndim == 3:
        return decoded[:, :, 0]
    return decoded


def encode_binary_mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask to a JSON-safe COCO RLE dict (`counts` as str)."""
    arr = np.asarray(mask)
    while arr.ndim > 2:
        arr = arr[0]
    arr = (arr > 0).astype(np.uint8)
    rle = mask_util.encode(np.asfortranarray(arr))
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("utf-8")
    return {"size": list(rle.get("size", arr.shape)), "counts": rle["counts"]}
