"""Wiring the real SAM3 and Claude implementations into the stage graph.

Everything else in this package is testable without a GPU or an API key because
the model- and SAM3-dependent steps are injected. This module is where those
injections are actually constructed, so it is the one place that needs the
`[sam3]` extra, CUDA and a key.

Kept deliberately thin: it builds callables and hands them to
`pipeline.frame.process_frame`. Any logic that grows here belongs in a tested
module instead.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..clickengine.parsing import parse_frame_validity
from ..config import RunConfig
from ..pipeline.frame import FrameOutcome, FrameStages, process_frame
from ..pipeline.loading import load_firstpass, load_initial_masks, write_json
from ..rle import encode_binary_mask_to_rle

SYSTEM_CANDIDATE = (
    "You are judging SAM3 segmentation candidates for one marine organism. "
    "Answer only with the requested tag."
)
SYSTEM_VERIFY = (
    "You are judging whether one segmentation mask is a tight, complete, "
    "single-identity mask of a real marine organism. Answer only with the "
    "requested tag."
)


class MissingExtraError(RuntimeError):
    """Raised when the SAM3 extra or an API key is absent."""


def _require_sam3() -> Any:
    """Check SAM3 is actually importable, then return the service builder.

    Checking the `sam3` package directly matters: `sam3svc.service` imports
    cleanly without it, because `build_sam3_service` defers its sam3 import to
    call time. Guarding on the wrapper would pass here and fail later, deep
    inside a run.
    """
    import importlib.util  # noqa: PLC0415

    missing = [name for name in ("sam3", "torch") if importlib.util.find_spec(name) is None]
    if missing:
        raise MissingExtraError(
            f"{' and '.join(missing)} not installed. Install the extra:\n"
            "    pip install -e '.[dev,sam3]'"
        )
    from ..sam3svc.service import build_sam3_service  # noqa: PLC0415

    return build_sam3_service


def _require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingExtraError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )


def read_frame(video_path: Path, frame_index: int) -> tuple[np.ndarray, float]:
    """Decode one frame and the video's frame rate."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
    return frame, fps


def resolve_path(value: str, roots: dict[str, str] | None = None) -> Path:
    """Expand ${MAL_*} placeholders in a manifest path from the environment."""
    roots = roots or {}
    resolved = str(value)
    for name in ("MAL_PROJECT_ROOT", "MAL_SEATUBE_ROOT", "MAL_RUNS_ROOT", "MAL_VIDEO_ROOT"):
        placeholder = "${" + name + "}"
        if placeholder in resolved:
            root = roots.get(name) or os.environ.get(name)
            if not root:
                raise MissingExtraError(
                    f"{value!r} refers to {placeholder} but {name} is not set. "
                    "See .env.example."
                )
            resolved = resolved.replace(placeholder, root)
    return Path(resolved).expanduser()


def build_frame_processor(
    config: RunConfig,
) -> Callable[[dict[str, Any], Path], FrameOutcome]:
    """Return a `process(record, frame_dir)` for `pipeline.graph.run_manifest`."""
    build_sam3_service = _require_sam3()
    _require_api_key()
    service = build_sam3_service()

    def process(record: dict[str, Any], frame_dir: Path) -> FrameOutcome:
        frame_id = str(record["id"])
        video_path = resolve_path(str(record["video"]))
        frame, fps = read_frame(video_path, int(record["frame_index"]))
        height, width = frame.shape[:2]

        known: list[dict[str, Any]] = []
        if not config.firstpass_skipped and config.firstpass_root is not None:
            known, _ = load_firstpass(
                Path(config.firstpass_root) / frame_id, config.models.firstpass
            )
        known.extend(
            load_initial_masks(
                Path(config.initial_mask_root) if config.initial_mask_root else None,
                frame_id,
                (height, width),
            )
        )

        cv2.imwrite(str(frame_dir / "target.png"), frame)
        config.write(frame_dir / "run_config.json")

        stages = _build_stages(
            config, service=service, frame_dir=frame_dir, fps=fps,
            video_path=video_path, frame_index=int(record["frame_index"]),
        )
        outcome = process_frame(
            frame_id,
            frame,
            stages=stages,
            known_masks=known,
            pass_count=config.clicks.mask_guided_passes,
            border_scan=config.clicks.border_scan,
            click_dedup_px=config.clicks.click_dedup_px,
        )

        write_json(
            frame_dir / "final_masks_rle.json",
            {
                "frame_size_hw": [height, width],
                "masks": [
                    encode_binary_mask_to_rle(np.asarray(entry["mask"]).astype(bool))
                    for entry in outcome.accepted
                ],
            },
        )
        write_json(frame_dir / "summary.json", outcome.summary())
        return outcome

    return process


def _tally(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _answer_json(text: str | None) -> dict[str, Any]:
    """Parse the last <answer>{...}</answer> payload, or {}."""
    import json  # noqa: PLC0415
    import re  # noqa: PLC0415

    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text or "", re.DOTALL)
    if not matches:
        return {}
    try:
        payload = json.loads(matches[-1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_stages(
    config: RunConfig,
    *,
    service: Any,
    frame_dir: Path,
    fps: float,
    video_path: Path,
    frame_index: int,
) -> FrameStages:
    """Construct the injected stages against the real SAM3 and Claude."""
    from ..clickengine.crop import mask_crop_geom  # noqa: PLC0415
    from ..clickengine.discovery import build_content  # noqa: PLC0415
    from ..clickengine.generate import refine_group  # noqa: PLC0415
    from ..clickengine.loop import response_token_budget  # noqa: PLC0415
    from ..clickengine.maskgen import hybrid_policy  # noqa: PLC0415
    from ..clickengine.parsing import parse_creature_click_groups  # noqa: PLC0415
    from ..clickengine.recovery import run_repair_rounds  # noqa: PLC0415
    from ..clickengine.verify_batch import filter_by_confidence, verify_masks  # noqa: PLC0415
    from ..geometry import mask_level_nms  # noqa: PLC0415
    from ..llm.claude import send_claude_request  # noqa: PLC0415
    from ..prompts import load as load_prompt  # noqa: PLC0415
    from ..sam3svc.zoom import predict_on_crop  # noqa: PLC0415
    from ..viz.crops import (  # noqa: PLC0415  # noqa: PLC0415
        default_upscale,
        render_binary_mask_crop,
        render_candidate_sheet,
        render_mask_crop,
    )
    from ..viz.views import extract_reference_frames, render_discovery_views  # noqa: PLC0415

    def ask(messages: list[dict[str, Any]], model: str) -> str | None:
        return send_claude_request(
            messages,
            model=model,
            effort=config.models.effort or None,
            max_tokens=response_token_budget(model, 1024),
        )

    def screen_quality(frame: np.ndarray) -> str | None:
        path = frame_dir / "quality_screen.png"
        cv2.imwrite(str(path), frame)
        response = send_claude_request(
            [
                {
                    "role": "system",
                    "content": load_prompt("som_frame_quality", config.profile),
                },
                {"role": "user", "content": [{"type": "image", "image": str(path)}]},
            ],
            model=config.models.quality_screen,
            effort=config.models.effort or None,
        )
        verdict = parse_frame_validity(response or "")
        return None if verdict != "corrupted" else "quality_screen_corrupted"

    def discover(*, frame, known_masks, region, pass_index, border_scan):
        pass_dir = frame_dir / f"pass_{pass_index}"
        left, right, top, bottom, label = region
        whole_frame = (left, right, top, bottom) == (0.0, 1.0, 0.0, 1.0)

        views = render_discovery_views(
            frame,
            known_masks,
            pass_dir / "views",
            focus_region=None if whole_frame else (left, right, top, bottom),
        )
        references = extract_reference_frames(
            video_path, frame_index, list(config.clicks.temporal_offsets),
            pass_dir / "refs",
        )
        instruction = f"Focus this pass on {label}. "
        if border_scan:
            instruction += (
                "Audit the frame border specifically: life clipped by the edge is "
                "routinely missed. "
            )
        content = build_content(
            view_paths=views, neighbour_paths=references, pass_instruction=instruction
        )
        response = ask(
            [
                {
                    "role": "system",
                    "content": load_prompt("som_click_discovery", config.profile),
                },
                {"role": "user", "content": content},
            ],
            config.models.click,
        )
        (pass_dir / "discovery_response.txt").write_text(response or "<none>")
        return parse_creature_click_groups(response or "")

    def recover(*, groups, known_masks, pass_index):
        pass_dir = frame_dir / f"pass_{pass_index}"
        stage_dir = pass_dir / "recovery"
        stage_dir.mkdir(parents=True, exist_ok=True)
        frame = cv2.imread(str(frame_dir / "target.png"))
        height, width = frame.shape[:2]

        def predict(clicks: list[dict[str, Any]]):
            coords = np.array(
                [[c["x"] * width, c["y"] * height] for c in clicks], dtype=np.float32
            )
            labels = np.array([int(c["label"]) for c in clicks], dtype=np.int64)
            return service.raw_predict(
                frame, point_coords=coords, point_labels=labels, multimask=True
            )

        def make_judge(group_id: int):
            def judge(*, masks, scores, clicks, attempt, iteration, budget_reached):
                geom = mask_crop_geom(masks[0], clicks, width, height, 0.30)
                sheet = render_candidate_sheet(
                    frame, masks, clicks, geom,
                    stage_dir / f"id{group_id}_a{attempt}_it{iteration}.png",
                )
                suffix = (
                    " The click budget is spent; choose good or reject."
                    if budget_reached else ""
                )
                response = ask(
                    [
                        {"role": "system", "content": SYSTEM_CANDIDATE},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": sheet},
                                {
                                    "type": "text",
                                    "text": load_prompt("candidate_selection") + suffix,
                                },
                            ],
                        },
                    ],
                    config.models.click,
                )
                (stage_dir / f"id{group_id}_a{attempt}_it{iteration}_judge.txt").write_text(
                    response or "<none>"
                )
                return _answer_json(response)
            return judge

        def zoom_predict_for(clicks_seed: list[dict[str, Any]]):
            """SAM3 on an upscaled crop around the seed, mapped back to full frame."""
            geom = mask_crop_geom(
                np.zeros((height, width), dtype=bool), clicks_seed, width, height,
                config.clicks.zoom_crop_frac,
            )
            upscale = default_upscale(geom[2], geom[3])

            def predict_zoom(clicks: list[dict[str, Any]]):
                def run(image, coords, labels):
                    return service.raw_predict(
                        image, point_coords=coords, point_labels=labels, multimask=True
                    )
                return predict_on_crop(frame, clicks, geom, upscale, run)

            return predict_zoom

        def generate_for(group: dict[str, Any]):
            group_id = int(group.get("id", 0))

            def full_frame():
                return refine_group(
                    group, predict=predict, judge=make_judge(group_id),
                    width=width, height=height, max_clicks=5, strict_quality=True,
                )

            def zoom():
                # Runs only on a seed the full-frame pass already worked from, so
                # it never abandons: it always returns its best body-covering mask.
                return refine_group(
                    group, predict=zoom_predict_for(group["clicks"]),
                    judge=make_judge(group_id), width=width, height=height,
                    max_clicks=4, max_attempts=1, max_area_frac=0.85,
                    strict_quality=False, never_abandon=True,
                )

            if config.clicks.mask_generator == "zoom":
                return zoom()
            if config.clicks.mask_generator == "mm":
                return full_frame()
            return hybrid_policy(full_frame, zoom)

        generated: list[dict[str, Any]] = []
        for group in groups:
            result, trace = generate_for(group)
            result["trace"] = trace
            result["seed_click"] = next(
                (c for c in group["clicks"] if int(c.get("label", 1)) == 1), None
            )
            generated.append(result)

        def make_verify(round_number: int):
            def judge_mask(result: dict[str, Any]) -> dict[str, Any]:
                mask = np.asarray(result["mask"]).astype(bool)
                clicks = result.get("clicks_used") or []
                geom = mask_crop_geom(mask, clicks, width, height, 0.22)
                upscale = default_upscale(geom[2], geom[3])
                tag = f"r{round_number}_id{result.get('creature_id', 0)}"
                overlay = render_mask_crop(
                    frame, mask, clicks, geom, stage_dir / f"{tag}.png", upscale
                )
                binary = render_binary_mask_crop(
                    mask, geom, stage_dir / f"{tag}_binary.png", upscale
                )
                response = ask(
                    [
                        {"role": "system", "content": SYSTEM_VERIFY},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": overlay},
                                {"type": "image", "image": binary},
                                {
                                    "type": "text",
                                    "text": (
                                        f"Target: {result.get('description', '')!r}. "
                                        + load_prompt("mask_verification")
                                    ),
                                },
                            ],
                        },
                    ],
                    config.models.verify,
                )
                (stage_dir / f"{tag}_verify.txt").write_text(response or "<none>")
                return _answer_json(response)
            return judge_mask

        kept, rejected = verify_masks(
            generated, judge=make_verify(0), strict_identity=True
        )
        n_verify_kept, n_verify_dropped = len(kept), len(rejected)

        # A rejection is only repairable if the verifier supplied an actionable
        # click. Recording the split makes a starved repair loop visible: a run
        # where nothing is repairable looks identical to one where repair simply
        # never helps.
        rejection_reasons = {
            "with_repair_click": sum(
                1 for r in rejected if r.get("mask_quality_repair_click")
            ),
            "no_repair_click": sum(
                1 for r in rejected if not r.get("mask_quality_repair_click")
            ),
            "failures": _tally(str(r.get("mask_quality_failure")) for r in rejected),
        }

        def regenerate(item, repair_click):
            clicks = [dict(c) for c in (item.get("clicks_used") or [])]
            clicks.append(dict(repair_click))
            result, _ = refine_group(
                {"id": item.get("creature_id", 0),
                 "description": item.get("description", ""),
                 "clicks": clicks},
                predict=predict,
                judge=make_judge(int(item.get("creature_id", 0))),
                width=width, height=height, max_clicks=5, strict_quality=True,
            )
            return result

        repair = run_repair_rounds(
            rejected,
            regenerate=regenerate,
            verify=lambda results, n: verify_masks(
                results, judge=make_verify(n), strict_identity=True
            ),
            max_repair_rounds=config.clicks.max_repair_rounds,
        )
        kept.extend(repair["recovered"])

        kept = [r for r in kept if np.asarray(r["mask"]).astype(bool).any()]
        kept, nms_removed = mask_level_nms(kept)
        confident, low = filter_by_confidence(kept, config.clicks.min_recovery_confidence)

        return {
            "recovered": confident,
            "low_confidence": low,
            "n_generated": len(generated),
            "n_verify_kept": n_verify_kept,
            "n_verify_dropped": n_verify_dropped,
            "rejection_reasons": rejection_reasons,
            "n_repair_recovered": repair["repaired"],
            "mask_nms_removed": nms_removed,
            "hit_round_cap": repair["hit_round_cap"],
            "repair_rounds": repair["rounds_run"],
        }

    return FrameStages(screen_quality=screen_quality, discover=discover, recover=recover)
