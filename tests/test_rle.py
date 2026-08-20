"""RLE round-trips, exercised against real masks captured from a GPU run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from marine_autolabel.rle import decode_rle_to_mask, encode_binary_mask_to_rle

# Defined here rather than imported from conftest: parametrize runs at collection
# time, and `tests/` is deliberately not a package.
FIXTURES = Path(__file__).parent / "fixtures"
FINAL_MASKS = sorted((FIXTURES / "run").glob("final_masks_rle.json"))


@pytest.mark.parametrize("path", FINAL_MASKS, ids=lambda p: p.name[:24])
def test_decodes_real_masks_to_the_declared_frame_size(path):
    payload = json.loads(path.read_text())
    height, width = payload["frame_size_hw"]
    masks = payload["masks"]
    assert masks, "fixture should carry at least one mask"

    for rle in masks:
        mask = decode_rle_to_mask(rle, height, width)
        assert mask.shape == (height, width)
        assert mask.any(), "a stored accepted mask should not be empty"


@pytest.mark.parametrize("path", FINAL_MASKS, ids=lambda p: p.name[:24])
def test_round_trip_is_lossless_on_real_masks(path):
    payload = json.loads(path.read_text())
    height, width = payload["frame_size_hw"]

    for rle in payload["masks"]:
        original = decode_rle_to_mask(rle, height, width).astype(bool)
        again = decode_rle_to_mask(encode_binary_mask_to_rle(original), height, width)
        assert np.array_equal(original, again.astype(bool))


def test_encode_emits_json_safe_counts():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 3:7] = True
    rle = encode_binary_mask_to_rle(mask)
    assert isinstance(rle["counts"], str), "counts must be str so the run file is JSON-serialisable"
    assert rle["size"] == [8, 8]
    json.dumps(rle)  # must not raise


def test_decode_accepts_bare_string_and_bytes_counts():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 1:4] = True
    rle = encode_binary_mask_to_rle(mask)

    from_dict = decode_rle_to_mask(rle, 6, 6)
    from_str = decode_rle_to_mask(rle["counts"], 6, 6)
    from_bytes = decode_rle_to_mask(
        {"counts": rle["counts"].encode("utf-8"), "size": [6, 6]}, 6, 6
    )
    assert np.array_equal(from_dict, from_str)
    assert np.array_equal(from_dict, from_bytes)


def test_decode_does_not_mutate_its_input():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    rle = encode_binary_mask_to_rle(mask)
    before = dict(rle)
    decode_rle_to_mask(rle, 4, 4)
    assert rle == before, "decode must not turn the caller's str counts into bytes"


def test_encode_squeezes_leading_dimensions():
    mask = np.zeros((1, 1, 5, 5), dtype=bool)
    mask[0, 0, 1:3, 1:3] = True
    rle = encode_binary_mask_to_rle(mask)
    assert rle["size"] == [5, 5]
