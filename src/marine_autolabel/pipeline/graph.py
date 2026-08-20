"""Running a manifest of frames, resumably.

Replaces the frame loop in `run_presentation_custom_flow.main`.

Resumability is the point. These runs take minutes per frame and have been
interrupted repeatedly by dropped connections and sleeping laptops, so a re-run
must skip frames that completed cleanly and retry frames whose model calls
failed. A frame that finished *with* a failed call is not cached -- its results
are wrong rather than incomplete.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..config import RunConfig
from .frame import FrameOutcome
from .loading import write_json
from .resume import is_frame_reusable

RESULT_NAME = "final_masks_rle.json"


def run_manifest(
    frames: Iterable[dict[str, Any]],
    config: RunConfig,
    *,
    process: Callable[[dict[str, Any], Path], FrameOutcome],
    resume: bool = True,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Process each frame in the manifest, skipping ones already done.

    `process(record, frame_dir)` runs one frame. Failures are contained: a frame
    that raises is recorded and the run continues, because losing a whole
    fan-out to one bad frame is the expensive outcome.
    """
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    def emit(event: str, payload: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event, payload)

    selected = list(frames)
    if config.frame_ids:
        wanted = set(config.frame_ids)
        selected = [record for record in selected if str(record["id"]) in wanted]

    results: list[dict[str, Any]] = []
    for record in selected:
        frame_id = str(record["id"])
        frame_dir = output_root / frame_id

        if resume and is_frame_reusable(frame_dir, result_name=RESULT_NAME):
            emit("reused", {"frame_id": frame_id})
            results.append({"frame_id": frame_id, "status": "reused"})
            continue

        frame_dir.mkdir(parents=True, exist_ok=True)
        emit("start", {"frame_id": frame_id})
        try:
            outcome = process(record, frame_dir)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not end the run
            emit("failed", {"frame_id": frame_id, "error": f"{type(exc).__name__}: {exc}"})
            results.append(
                {
                    "frame_id": frame_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        summary = outcome.summary()
        summary["status"] = "skipped" if outcome.skipped_reason else "done"
        results.append(summary)
        emit("done", summary)

    run_summary = {
        "config": config.to_dict(),
        "n_frames": len(selected),
        "n_done": sum(1 for r in results if r.get("status") == "done"),
        "n_reused": sum(1 for r in results if r.get("status") == "reused"),
        "n_skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "n_failed": sum(1 for r in results if r.get("status") == "failed"),
        "frames": results,
    }
    write_json(output_root / "summary.json", run_summary)
    return run_summary


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Read a benchmark manifest and return its frame records."""
    from .loading import read_json

    doc = read_json(Path(path))
    frames = doc.get("frames") or doc.get("samples")
    if not isinstance(frames, list):
        raise ValueError(f"manifest {path} has no 'frames' or 'samples' list")
    missing = [i for i, record in enumerate(frames) if "id" not in record]
    if missing:
        raise ValueError(f"manifest {path}: records at {missing} have no 'id'")
    return frames
