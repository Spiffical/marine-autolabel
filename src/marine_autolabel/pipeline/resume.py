"""Deciding whether a finished frame can be reused.

Ported from `run_presentation_custom_flow._api_failed`.

Long fan-outs get interrupted -- a dropped connection, a sleeping laptop, a
rate limit that outlasted the retries. When that happens the model's recorded
response is the sentinel `<none>`, and the frame's results are wrong rather than
merely incomplete. Caching such a frame would bake the failure in permanently,
so a frame is reusable only if no recorded response failed.
"""
from __future__ import annotations

from pathlib import Path

API_FAILURE_SENTINEL = "<none>"


def api_failed(frame_dir: Path) -> bool:
    """Did any recorded model response in this frame fail?

    Scans every `.txt` artefact under the directory. Unreadable files are
    skipped rather than treated as failures, since a file that cannot be read
    is not evidence either way.
    """
    for path in Path(frame_dir).rglob("*.txt"):
        try:
            if API_FAILURE_SENTINEL in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def is_frame_reusable(frame_dir: Path, *, result_name: str = "final_masks_rle.json") -> bool:
    """Can this frame be skipped on a re-run?

    Requires both a completed result and a clean set of responses. A frame that
    finished but contains a failed call is retried, which is what makes an
    interrupted fan-out resumable without silently keeping bad frames.
    """
    frame_dir = Path(frame_dir)
    if not (frame_dir / result_name).exists():
        return False
    return not api_failed(frame_dir)
