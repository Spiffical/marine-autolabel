#!/usr/bin/env python3
"""Drive the pipeline with an EXTERNAL judge serving every model call.

Runs on the GPU host. Each subcommand advances one stage and then stops,
leaving `*_request` manifests (images + prompt) for whoever is serving the
model -- a human, an agent, or the API. Responses are dropped next to the
requests as `*_response.txt`, and the next subcommand applies them using the
SAME repo functions the live pipeline uses (parsers, geometry, judgement
prompts, verify semantics, NMS, confidence filter).

This exists so the full loop can run and be improved at zero API cost, with
the serving model swapped in only at the end.

Stages per pass:
    views          render discovery views + refs; write the discovery request
    groups         parse the discovery response into click groups
    gen            SAM3 for every active group; write judge requests
    judge-apply    apply judge verdicts (good/add/reject/abandon); loop to gen
    verify-mat     render verify materials; write verify requests
    verify-apply   apply verdicts; build repair round or finish the pass
    accept         NMS + confidence; merge into the accepted set; composite
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from marine_autolabel.clickengine.crop import mask_crop_geom
from marine_autolabel.clickengine.discovery import build_discovery_prompt
from marine_autolabel.clickengine.judgement import (
    DUPLICATE_FEEDBACK,
    build_judge_prompt,
    build_verify_prompt,
)
from marine_autolabel.clickengine.parsing import parse_creature_click_groups
from marine_autolabel.clickengine.recovery import is_actionable_repair_click
from marine_autolabel.clickengine.sweep import (
    dedup_proposals,
    discovery_focus_region,
    filter_prior_attempt_groups,
    first_positive_click,
    should_run_border_scan,
)
from marine_autolabel.clickengine.verify import (
    accept_mask_verdict,
    mask_quality_repair_click,
)
from marine_autolabel.clickengine.verify_batch import (
    KNOWN_FAILURES,
    coerce_confidence,
)
from marine_autolabel.geometry import (
    clean_candidate_components,
    duplicate_click,
    mask_level_nms,
    smallest_valid,
)
from marine_autolabel.rle import decode_rle_to_mask, encode_binary_mask_to_rle
from marine_autolabel.viz.crops import (
    default_upscale,
    render_binary_mask_crop,
    render_candidate_sheet,
    render_fullframe_candidate,
    render_mask_crop,
    stack_review_sheet,
)
from marine_autolabel.sam3svc.text import (
    build_planner_prompt,
    in_exclusion_region,
    select_prompt_specs,
)
from marine_autolabel.viz.views import extract_reference_frames, render_discovery_views

MAX_CLICKS = 5
MAX_AREA_FRAC = 0.60
MIN_CONFIDENCE = 0.5
MAX_REPAIR_ROUNDS = 2
JSON = dict


def _read(path: Path) -> JSON:
    return json.loads(path.read_text())


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1))


def _answer_json(text: str) -> JSON:
    import re
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text or "", re.DOTALL)
    if not matches:
        return {}
    try:
        payload = json.loads(matches[-1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _state(root: Path) -> JSON:
    path = root / "state.json"
    if path.exists():
        return _read(path)
    return {"accepted": [], "prior_attempts": [], "passes": []}


def _accepted_masks(state: JSON, h: int, w: int) -> list[dict]:
    return [
        {"mask": decode_rle_to_mask(item["rle"], h, w).astype(bool), **{
            k: v for k, v in item.items() if k != "rle"}}
        for item in state["accepted"]
    ]


def _frame(root: Path) -> np.ndarray:
    return cv2.imread(str(root / "target.png"))


def cmd_views(root: Path, pass_idx: int, pass_count: int, video: str,
              frame_index: int, offsets: list[int]) -> None:
    state = _state(root)
    frame = _frame(root)
    h, w = frame.shape[:2]
    region = discovery_focus_region(pass_idx, pass_count)
    border = should_run_border_scan(
        "last", convergence_mode=pass_count == 0, pass_index=pass_idx,
        requested_passes=pass_count)
    pdir = root / f"pass_{pass_idx}"
    whole = region[:4] == (0.0, 1.0, 0.0, 1.0)
    views = render_discovery_views(
        frame, _accepted_masks(state, h, w), pdir / "views",
        focus_region=None if whole else region[:4], with_clahe=True)
    refs = extract_reference_frames(Path(video), frame_index, offsets, pdir / "refs")
    instruction = f"Focus this pass on {region[4]}. "
    if border:
        instruction += ("Audit the frame border specifically: life clipped by the "
                        "edge is routinely missed. ")
    order = [k for k in ("raw", "grid", "strong", "outline", "clahe",
                         "focus_raw", "focus_strong") if k in views]
    _write(pdir / "discovery_request.json", {
        "images": [views[k] for k in order] + refs,
        "prompt_file": str(pdir / "discovery_prompt.txt"),
    })
    (pdir / "discovery_prompt.txt").write_text(build_discovery_prompt(
        pass_instruction=instruction,
        has_focus_crops="focus_raw" in views,
        has_clahe="clahe" in views))
    print(f"pass {pass_idx}: region={region[4]!r} border={border} "
          f"views={len(order)} refs={len(refs)}")


def cmd_groups(root: Path, pass_idx: int) -> None:
    state = _state(root)
    frame = _frame(root)
    h, w = frame.shape[:2]
    pdir = root / f"pass_{pass_idx}"
    response = (pdir / "discovery_response.txt").read_text()
    proposed = parse_creature_click_groups(response)
    fresh, skipped = filter_prior_attempt_groups(proposed, state["prior_attempts"])
    deduped, removed = dedup_proposals(fresh, w, h, px=40)
    for group in deduped:
        seed = first_positive_click(group)
        if seed:
            state["prior_attempts"].append(
                {"click": seed, "description": group.get("description", "")})
    groups = {
        str(g["id"]): {
            "id": g["id"], "description": g.get("description", ""),
            "clicks": g["clicks"], "status": "active", "iteration": 0,
            "duplicate_retries": 0, "repair_round": 0, "repair_history": [],
        } for g in deduped
    }
    _write(pdir / "groups.json", groups)
    _write(root / "state.json", state)
    print(f"proposed={len(proposed)} repeat_skipped={len(skipped)} "
          f"dedup_removed={removed} active={len(groups)}")


def _predict(service, frame, clicks, w, h):
    coords = np.array([[c["x"] * w, c["y"] * h] for c in clicks], dtype=np.float32)
    labels = np.array([int(c["label"]) for c in clicks], dtype=np.int64)
    return service.raw_predict(frame, point_coords=coords, point_labels=labels,
                               multimask=True)


def cmd_gen(root: Path, pass_idx: int) -> None:
    from marine_autolabel.sam3svc.service import build_sam3_service
    service = build_sam3_service()
    frame = _frame(root)
    h, w = frame.shape[:2]
    pdir = root / f"pass_{pass_idx}"
    groups = _read(pdir / "groups.json")
    requests = []
    for gid, group in groups.items():
        if group["status"] != "active":
            continue
        masks, scores = _predict(service, frame, group["clicks"], w, h)
        masks = clean_candidate_components(masks, group["clicks"])
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        valid = areas <= MAX_AREA_FRAC * h * w
        tag = f"g{gid}_r{group['repair_round']}_it{group['iteration']}"
        gdir = pdir / "gen"
        gdir.mkdir(parents=True, exist_ok=True)
        if not valid.any():
            group["status"] = "abandoned"
            group["note"] = "no_valid_band"
            continue
        best = int(np.argmax(np.where(valid, scores, -np.inf)))
        geom = mask_crop_geom(masks[best], group["clicks"], w, h, 0.30)
        render_candidate_sheet(frame, masks, group["clicks"], geom,
                               gdir / f"{tag}_sheet.png")
        np.savez_compressed(gdir / f"{tag}_masks.npz", masks=masks, scores=scores,
                            valid=valid, geom=np.array(geom))
        budget = len(group["clicks"]) >= MAX_CLICKS
        prompt = build_judge_prompt(
            group["description"],
            duplicate_feedback=DUPLICATE_FEEDBACK if group["duplicate_retries"] else "",
            budget_reached=budget)
        (gdir / f"{tag}_judge_prompt.txt").write_text(prompt)
        requests.append({
            "gid": gid, "tag": tag,
            "images": [str(root / "target.png"), str(gdir / f"{tag}_sheet.png")],
            "prompt_file": str(gdir / f"{tag}_judge_prompt.txt"),
            "response_file": str(gdir / f"{tag}_judge_response.txt"),
        })
    _write(pdir / "groups.json", groups)
    _write(pdir / "judge_requests.json", requests)
    print(f"judge requests: {len(requests)}; "
          f"abandoned_now: {[g for g in groups.values() if g.get('note')]}")


def cmd_judge_apply(root: Path, pass_idx: int) -> None:
    frame = _frame(root)
    h, w = frame.shape[:2]
    pdir = root / f"pass_{pass_idx}"
    groups = _read(pdir / "groups.json")
    requests = _read(pdir / "judge_requests.json")
    still_active = 0
    for req in requests:
        group = groups[req["gid"]]
        data = np.load(pdir / "gen" / f"{req['tag']}_masks.npz")
        masks, scores, valid = data["masks"], data["scores"], data["valid"]
        left, top, cw, ch = (int(v) for v in data["geom"])
        answer = _answer_json(Path(req["response_file"]).read_text())
        verdict = str(answer.get("verdict", ""))
        if verdict == "good":
            index = answer.get("choice", answer.get("index"))
            if not isinstance(index, int) or not (0 <= index < len(masks)) \
                    or not valid[index]:
                index = smallest_valid(masks, valid)
            group["status"] = "picked"
            group["picked_rle"] = encode_binary_mask_to_rle(masks[index])
            group["score"] = float(scores[index])
        elif verdict == "abandon":
            group["status"] = "abandoned"
        elif verdict == "reject":
            group["status"] = "abandoned"           # max_attempts=1, no reseed
            group["note"] = "judge_reject"
        elif verdict == "add" and isinstance(answer.get("click"), dict) \
                and len(group["clicks"]) < MAX_CLICKS:
            click = answer["click"]
            candidate = {
                "x": (left + float(click["x"]) * cw) / w,
                "y": (top + float(click["y"]) * ch) / h,
                "label": int(click.get("label", 1)),
            }
            if duplicate_click(group["clicks"], candidate):
                group["duplicate_retries"] += 1
                if group["duplicate_retries"] >= 2:
                    group["status"] = "abandoned"
                    group["note"] = "duplicate_no_progress"
                else:
                    still_active += 1
            else:
                group["clicks"].append(candidate)
                group["iteration"] += 1
                still_active += 1
        else:
            # unusable verdict under strict quality -> abandoned
            group["status"] = "abandoned"
            group["note"] = f"unusable_verdict:{verdict!r}"
    _write(pdir / "groups.json", groups)
    print(f"picked={sum(g['status'] == 'picked' for g in groups.values())} "
          f"abandoned={sum(g['status'] == 'abandoned' for g in groups.values())} "
          f"still_active={still_active}")


def cmd_verify_mat(root: Path, pass_idx: int, round_n: int) -> None:
    frame = _frame(root)
    h, w = frame.shape[:2]
    pdir = root / f"pass_{pass_idx}"
    groups = _read(pdir / "groups.json")
    vdir = pdir / "verify"
    requests = []
    for gid, group in groups.items():
        if group["status"] != "picked" or group.get("verify_round", -1) >= round_n:
            continue
        mask = decode_rle_to_mask(group["picked_rle"], h, w).astype(bool)
        if not mask.any():
            group["status"] = "abandoned"
            continue
        tag = f"g{gid}_vr{round_n}"
        vdir.mkdir(parents=True, exist_ok=True)
        geom = mask_crop_geom(mask, group["clicks"], w, h, 0.22)
        up = default_upscale(geom[2], geom[3])
        overlay = render_mask_crop(frame, mask, group["clicks"], geom,
                                   vdir / f"{tag}_crop.png", up)
        binary = render_binary_mask_crop(mask, geom, vdir / f"{tag}_binary.png", up)
        stack_review_sheet(overlay, binary, vdir / f"{tag}_review.png")
        render_fullframe_candidate(frame, mask, vdir / f"{tag}_context.png")
        (vdir / f"{tag}_prompt.txt").write_text(build_verify_prompt(
            group["description"], mask, w, h,
            repair_history=group["repair_history"] or None,
            allow_all_life=True, occlusion_addendum=True))
        group["verify_round"] = round_n
        requests.append({
            "gid": gid, "tag": tag,
            "images": [str(root / "target.png"), str(vdir / f"{tag}_context.png"),
                       str(vdir / f"{tag}_review.png")],
            "prompt_file": str(vdir / f"{tag}_prompt.txt"),
            "response_file": str(vdir / f"{tag}_response.txt"),
        })
    _write(pdir / "groups.json", groups)
    _write(pdir / "verify_requests.json", requests)
    print(f"verify requests (round {round_n}): {len(requests)}")


def cmd_verify_apply(root: Path, pass_idx: int, round_n: int) -> None:
    pdir = root / f"pass_{pass_idx}"
    groups = _read(pdir / "groups.json")
    requests = _read(pdir / "verify_requests.json")
    kept = dropped = repairable = 0
    for req in requests:
        group = groups[req["gid"]]
        answer = _answer_json(Path(req["response_file"]).read_text())
        confidence = coerce_confidence(answer.get("confidence"))
        keep = accept_mask_verdict(answer, strict_identity=True)
        failure = str(answer.get("failure", "")).strip().lower()
        group["confidence"] = confidence if confidence is not None else (
            0.75 if keep else 0.1)
        group["failure"] = failure if failure in KNOWN_FAILURES else None
        if keep:
            group["status"] = "verified"
            kept += 1
            continue
        single = answer.get("single_identity") is True
        repair = mask_quality_repair_click(answer)
        prior = [c for c in group["clicks"] if c.get("label") in (0, 1)]
        if (repair and group["repair_round"] < MAX_REPAIR_ROUNDS
                and is_actionable_repair_click(repair, prior)):
            group["repair_history"].append(
                {"round": group["repair_round"] + 1,
                 "failure": group["failure"] or "unknown", "click": repair})
            group["clicks"].append(repair)
            group["repair_round"] += 1
            group["iteration"] = 0
            group["duplicate_retries"] = 0
            group["status"] = "active"       # back through gen/judge
            repairable += 1
        elif group["failure"] == "fragment" and single:
            # Policy 2026-08-20: keep the visible extent, flagged partial.
            group["status"] = "partial"
            dropped += 1
        else:
            group["status"] = "dropped"
            dropped += 1
    _write(pdir / "groups.json", groups)
    partial = sum(1 for g in groups.values() if g["status"] == "partial")
    print(f"kept={kept} dropped={dropped} (partial-kept {partial}) "
          f"repair_pending={repairable}")


def cmd_zoom_gen(root: Path, pass_idx: int, gid: str) -> None:
    """Zoom-fallback regeneration for one abandoned group, as hybrid would run.

    Mirrors the production hybrid policy: the full-frame path abandoned or was
    judge-rejected, so re-segment on an upscaled crop around the clicks, where
    a small or camouflaged target fills the field of view. Proven live on a
    column SAM3 merged with a rock dome for three full-frame iterations and
    isolated at the first zoom attempt.
    """
    from marine_autolabel.sam3svc.service import build_sam3_service
    from marine_autolabel.sam3svc.zoom import predict_on_crop

    service = build_sam3_service()
    frame = _frame(root)
    h, w = frame.shape[:2]
    pdir = root / f"pass_{pass_idx}"
    groups = _read(pdir / "groups.json")
    group = groups[gid]

    geom = mask_crop_geom(np.zeros((h, w), bool), group["clicks"], w, h, 0.50)
    up = default_upscale(geom[2], geom[3])

    def run(image, coords, labels):
        return service.raw_predict(image, point_coords=coords,
                                   point_labels=labels, multimask=True)

    masks, scores = predict_on_crop(frame, group["clicks"], geom, up, run)
    masks = clean_candidate_components(masks, group["clicks"])
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    valid = areas <= 0.85 * h * w
    tag = f"g{gid}_zoom"
    gdir = pdir / "gen"
    best = int(np.argmax(np.where(valid, scores, -np.inf))) if valid.any() else 0
    sheet_geom = mask_crop_geom(masks[best], group["clicks"], w, h, 0.30)
    render_candidate_sheet(frame, masks, group["clicks"], sheet_geom,
                           gdir / f"{tag}_sheet.png")
    np.savez_compressed(gdir / f"{tag}_masks.npz", masks=masks, scores=scores,
                        valid=valid, geom=np.array(sheet_geom))
    (gdir / f"{tag}_judge_prompt.txt").write_text(
        build_judge_prompt(group["description"], budget_reached=True))
    print(f"zoom candidates for g{gid}: areas={areas.tolist()} "
          f"scores={np.round(scores, 3).tolist()}")


def cmd_accept(root: Path, pass_idx: int) -> None:
    state = _state(root)
    frame = _frame(root)
    h, w = frame.shape[:2]
    pdir = root / f"pass_{pass_idx}"
    groups = _read(pdir / "groups.json")
    results = []
    for gid, group in groups.items():
        if group["status"] not in ("verified", "partial"):
            continue
        mask = decode_rle_to_mask(group["picked_rle"], h, w).astype(bool)
        seed = first_positive_click({"clicks": group["clicks"]})
        results.append({"mask": mask, "seed_click": seed, "gid": gid,
                        "confidence": group.get("confidence", 0.0),
                        "partial": group["status"] == "partial",
                        "description": group["description"]})
    results = [r for r in results if r["mask"].any()]
    results, nms_removed = mask_level_nms(results)
    confident = [r for r in results
                 if r.get("partial") or r["confidence"] >= MIN_CONFIDENCE]
    for r in confident:
        state["accepted"].append({
            "rle": encode_binary_mask_to_rle(r["mask"]),
            "description": r["description"], "confidence": r["confidence"],
            "pass": pass_idx, "gid": r["gid"],
            **({"partial": True, "select_reason": "partial_fragment_keep"}
               if r.get("partial") else {}),
        })
    state["passes"].append({
        "pass": pass_idx,
        "n_groups": len(groups),
        "n_verified": len(results) + nms_removed,
        "n_nms_removed": nms_removed,
        "n_low_confidence": len(results) - len(confident),
        "n_accepted": len(confident),
        "statuses": {gid: g["status"] for gid, g in groups.items()},
    })
    _write(root / "state.json", state)
    out = frame.copy()
    colors = [(40, 220, 40), (210, 80, 220), (40, 190, 240), (230, 150, 40),
              (60, 80, 235), (220, 220, 50), (170, 80, 240)]
    for i, item in enumerate(state["accepted"]):
        mask = decode_rle_to_mask(item["rle"], h, w).astype(bool)
        c = colors[i % len(colors)]
        out[mask] = (0.4 * np.array(c) + 0.6 * out[mask]).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, c, 2)
        ys, xs = np.where(mask)
        if len(xs):
            for th, cc in ((4, (0, 0, 0)), (1, c)):
                cv2.putText(out, str(i + 1), (int(xs.mean()), int(ys.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, cc, th, cv2.LINE_AA)
    cv2.imwrite(str(root / f"composite_after_pass_{pass_idx}.png"), out)
    print(f"pass {pass_idx} accepted={len(confident)} "
          f"total={len(state['accepted'])} nms={nms_removed}")


def cmd_fp_plan(root: Path, frame_id: str, video: str, frame_index: int,
                candidates: list[str], visual_note: str,
                exclude: list[list[float]]) -> None:
    """First-pass stage 1: extract the frame, write the phrase-planner request."""
    root.mkdir(parents=True, exist_ok=True)
    target = root / "target.png"
    if not target.exists():
        capture = cv2.VideoCapture(video)
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {video}")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise RuntimeError(f"could not read frame {frame_index} from {video}")
        cv2.imwrite(str(target), frame)
    fdir = root / "firstpass"
    fdir.mkdir(parents=True, exist_ok=True)
    prompt = build_planner_prompt(frame_id, visual_note, candidates)
    (fdir / "planner_prompt.txt").write_text(prompt)
    _write(fdir / "config.json", {
        "frame_id": frame_id, "candidates": candidates,
        "visual_note": visual_note, "exclusion_regions": exclude,
    })
    _write(fdir / "planner_request.json", {
        "images": [str(target)],
        "prompt_file": str(fdir / "planner_prompt.txt"),
        "response_file": str(fdir / "planner_response.txt"),
    })
    print(f"fp-plan {frame_id}: candidates={len(candidates)} "
          f"exclusions={len(exclude)}")


def cmd_fp_gen(root: Path) -> None:
    """First-pass stage 2: SAM3 text masks for the selected phrases; seed state.

    Mirrors production: selected phrases are closed-vocabulary retrieval
    handles, the processor's built-in confidence threshold (0.5) gates the
    instances, exclusion regions drop overlay hits, NMS dedups across phrases,
    and survivors seed the accepted set exactly as `load_firstpass` masks do.
    """
    from marine_autolabel.sam3svc.service import as_pil_rgb, build_sam3_service
    import torch

    fdir = root / "firstpass"
    cfg = _read(fdir / "config.json")
    answer = _answer_json((fdir / "planner_response.txt").read_text())
    specs = select_prompt_specs(cfg["candidates"], answer)
    frame = _frame(root)
    h, w = frame.shape[:2]
    image = as_pil_rgb(frame)

    service = build_sam3_service()
    processor = service.processor
    instances = []
    per_phrase = {}
    for spec in specs:
        with torch.inference_mode():
            state_out = processor.set_image(image)
            state_out = processor.set_text_prompt(state=state_out, prompt=spec.text)
        masks, boxes, scores = state_out["masks"], state_out["boxes"], state_out["scores"]
        masks = (masks.squeeze(1).detach().float().cpu().numpy()
                 if torch.is_tensor(masks) else np.asarray(masks).squeeze(1))
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        boxes = (boxes.detach().float().cpu().numpy()
                 if torch.is_tensor(boxes) else np.asarray(boxes))
        boxes = boxes.reshape(-1, 4) if boxes.size else np.zeros((0, 4))
        scores = (scores.detach().float().cpu().numpy()
                  if torch.is_tensor(scores) else np.asarray(scores)).reshape(-1)
        kept = excluded = 0
        for i in range(masks.shape[0]):
            mask = masks[i].astype(bool)
            if not mask.any():
                continue
            box = boxes[i] if i < len(boxes) else None
            if box is not None and in_exclusion_region(
                    (box[0] / w, box[1] / h, box[2] / w, box[3] / h),
                    cfg["exclusion_regions"]):
                excluded += 1
                continue
            ys, xs = np.where(mask)
            instances.append({
                "mask": mask,
                "seed_click": {"x": float(xs.mean() / w), "y": float(ys.mean() / h)},
                "phrase": spec.text,
                "prob": float(scores[i]) if i < len(scores) else None,
            })
            kept += 1
        per_phrase[spec.text] = {"kept": kept, "excluded": excluded}
    instances, nms_removed = mask_level_nms(instances)

    _write(fdir / "frame_outputs_rle.json", {
        "frame_size_hw": [h, w],
        "frames": [{
            "out_binary_masks_rle": [encode_binary_mask_to_rle(it["mask"])
                                     for it in instances],
            "out_probs": [it["prob"] for it in instances],
            "out_boxes_xywh": [],
        }],
    })
    _write(fdir / "summary.json", {
        "model": "subagent-harness", "error_count": 0,
        "phrases": [s.text for s in specs], "per_phrase": per_phrase,
        "n_masks": len(instances), "nms_removed": nms_removed,
    })
    state = _state(root)
    for i, it in enumerate(instances):
        state["accepted"].append({
            "rle": encode_binary_mask_to_rle(it["mask"]),
            "description": f"firstpass:{it['phrase']}",
            "confidence": it["prob"] if it["prob"] is not None else 0.0,
            "pass": -1, "gid": f"fp{i}", "source": "firstpass",
        })
    _write(root / "state.json", state)
    out = frame.copy()
    for i, it in enumerate(instances):
        c = (40, 220, 40)
        out[it["mask"]] = (0.45 * np.array(c) + 0.55 * out[it["mask"]]).astype(np.uint8)
        cnts, _ = cv2.findContours(it["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, c, 2)
    cv2.imwrite(str(root / "composite_after_firstpass.png"), out)
    print(f"fp-gen: phrases={[s.text for s in specs]} per_phrase={per_phrase} "
          f"nms_removed={nms_removed} seeded={len(instances)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["views", "groups", "gen", "judge-apply",
                                        "verify-mat", "verify-apply", "accept",
                                        "zoom-gen", "fp-plan", "fp-gen"])
    parser.add_argument("root", type=Path)
    parser.add_argument("pass_idx", type=int)
    parser.add_argument("--pass-count", type=int, default=0)
    parser.add_argument("--video", default="")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--offsets", default="15,30")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--gid", default="")
    parser.add_argument("--frame-id", default="")
    parser.add_argument("--candidates", default="",
                        help="pipe-separated allowed phrases")
    parser.add_argument("--visual-note", default="")
    parser.add_argument("--exclude", default="",
                        help="semicolon-separated normalized xyxy regions, "
                             "each 'x0,y0,x1,y1'")
    a = parser.parse_args()
    if a.cmd == "views":
        cmd_views(a.root, a.pass_idx, a.pass_count, a.video, a.frame_index,
                  [int(x) for x in a.offsets.split(",") if x])
    elif a.cmd == "groups":
        cmd_groups(a.root, a.pass_idx)
    elif a.cmd == "gen":
        cmd_gen(a.root, a.pass_idx)
    elif a.cmd == "judge-apply":
        cmd_judge_apply(a.root, a.pass_idx)
    elif a.cmd == "verify-mat":
        cmd_verify_mat(a.root, a.pass_idx, a.round)
    elif a.cmd == "verify-apply":
        cmd_verify_apply(a.root, a.pass_idx, a.round)
    elif a.cmd == "accept":
        cmd_accept(a.root, a.pass_idx)
    elif a.cmd == "zoom-gen":
        cmd_zoom_gen(a.root, a.pass_idx, a.gid)
    elif a.cmd == "fp-plan":
        cmd_fp_plan(
            a.root, a.frame_id, a.video, a.frame_index,
            [p.strip() for p in a.candidates.split("|") if p.strip()],
            a.visual_note,
            [[float(v) for v in region.split(",")]
             for region in a.exclude.split(";") if region.strip()])
    elif a.cmd == "fp-gen":
        cmd_fp_gen(a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
