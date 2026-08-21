#!/usr/bin/env python3
"""Render standalone verification requests for every accepted mask of a run.

For each mask in a frame's final_masks_rle.json this writes the EXACT materials
the verify stage sends -- raw frame, full-frame candidate context, review sheet
-- plus the byte-faithful prompt, into case_NN/ directories.

Purpose: replaying verification OFFLINE, with any judge standing in for the
model (a human, another agent, or the API). This is how the 2026-08 materials
regression was diagnosed and the fix validated without paying for a live run.

Usage:
    python benchmarks/render_verify_replay.py <frame_run_dir> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

from marine_autolabel.clickengine.crop import mask_crop_geom
from marine_autolabel.clickengine.judgement import build_verify_prompt
from marine_autolabel.rle import decode_rle_to_mask
from marine_autolabel.viz.crops import (
    default_upscale,
    render_binary_mask_crop,
    render_fullframe_candidate,
    render_mask_crop,
    stack_review_sheet,
)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    run_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = json.loads((run_dir / "final_masks_rle.json").read_text())
    height, width = doc["frame_size_hw"]
    frame = cv2.imread(str(run_dir / "target.png"))
    cv2.imwrite(str(out_dir / "target.png"), frame)

    for index, rle in enumerate(doc["masks"], 1):
        mask = decode_rle_to_mask(rle, height, width).astype(bool)
        case = out_dir / f"case_{index:02d}"
        case.mkdir(exist_ok=True)
        geom = mask_crop_geom(mask, [], width, height, 0.22)
        upscale = default_upscale(geom[2], geom[3])
        overlay = render_mask_crop(frame, mask, [], geom, case / "crop.png", upscale)
        binary = render_binary_mask_crop(mask, geom, case / "binary.png", upscale)
        stack_review_sheet(overlay, binary, case / "review.png")
        render_fullframe_candidate(frame, mask, case / "context.png")
        (case / "prompt.txt").write_text(
            build_verify_prompt("", mask, width, height, allow_all_life=True)
        )
        print(f"case_{index:02d}: area={int(mask.sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
