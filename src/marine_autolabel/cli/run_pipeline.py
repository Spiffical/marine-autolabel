"""`mal-pipeline` -- run the click-recovery pipeline over a frame manifest.

A run is described by a config file, not by forty flags. The previous
orchestrator took 31 keyword arguments plus ~40 CLI options, which made runs
hard to reproduce and impossible to diff. Here the config file is the record,
and it is copied into the output directory alongside source provenance.

Flags exist only for what genuinely varies per invocation: which config, which
frames, and whether to resume.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..config import RunConfig
from ..pipeline.graph import load_manifest, run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mal-pipeline", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--config", required=True, type=Path, help="run config JSON")
    parser.add_argument(
        "--frame-id",
        action="append",
        dest="frame_ids",
        default=[],
        help="restrict to these frame ids (repeatable)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="reprocess frames that already completed cleanly",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the config and manifest, then report what would run",
    )
    return parser


def _report(event: str, payload: dict[str, Any]) -> None:
    frame_id = payload.get("frame_id", "?")
    if event == "start":
        print(f"[frame] {frame_id} ...", flush=True)
    elif event == "reused":
        print(f"[frame] {frame_id} reused", flush=True)
    elif event == "failed":
        print(f"[frame] {frame_id} FAILED: {payload.get('error')}", flush=True)
    elif event == "done":
        note = payload.get("skipped_reason") or f"{payload.get('n_accepted', 0)} masks"
        print(f"[frame] {frame_id} done ({note})", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = RunConfig.from_file(args.config)
    if args.frame_ids:
        config = type(config)(**{**config_as_kwargs(config), "frame_ids": tuple(args.frame_ids)})

    frames = load_manifest(config.manifest)
    selected = (
        [f for f in frames if str(f["id"]) in set(config.frame_ids)]
        if config.frame_ids
        else frames
    )

    if args.dry_run:
        print(f"config      : {args.config}")
        print(f"manifest    : {config.manifest} ({len(frames)} records)")
        print(f"output      : {config.output_dir}")
        print(f"models      : click={config.models.click} phrase={config.models.resolved_phrase()}")
        print(f"              firstpass={config.models.firstpass} effort={config.models.effort!r}")
        print(f"would run   : {[str(f['id']) for f in selected]}")
        return 0

    from ._wiring import build_frame_processor  # noqa: PLC0415 - needs the sam3 extra

    summary = run_manifest(
        frames,
        config,
        process=build_frame_processor(config),
        resume=not args.no_resume,
        on_event=_report,
    )
    print(
        f"\n{summary['n_done']} done, {summary['n_reused']} reused, "
        f"{summary['n_skipped']} skipped, {summary['n_failed']} failed"
    )
    return 1 if summary["n_failed"] else 0


def config_as_kwargs(config: RunConfig) -> dict[str, Any]:
    """Shallow field mapping, preserving the nested dataclass instances."""
    import dataclasses

    return {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}


if __name__ == "__main__":
    sys.exit(main())
