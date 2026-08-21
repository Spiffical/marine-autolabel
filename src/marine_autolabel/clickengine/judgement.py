"""The judgement prompts: candidate selection and mask verification.

Ported VERBATIM from `click_engine_probe.py` (`refine_group_mm`'s inline judge
text, `MM_JUDGE_FMT`, `verify_masks`'s inline rubric, `MASK_VERIFY_FMT`).

History note, because it is the reason this module exists: the first port
paraphrased these prompts down to their answer contracts. A production run then
rejected 31 of 51 real generated masks while accepting blobs on empty water --
the paraphrase had dropped the exact sentence defending against that
("Cyan/WHITE having a smooth, coherent silhouette is NOT evidence that an
organism exists...") along with the full-frame context images and the
deterministic geometry facts. The prompt text is tuned. It is not to be
improved in passing.
"""
from __future__ import annotations

from typing import Any

import numpy as np

MM_JUDGE_FMT = (
    'Output: brief reasoning then EXACTLY ONE trailing tag, one of:\n'
    '<answer>{"verdict": "good", "choice": <0|1|2>}</answer>'
    '  -- candidate #choice tightly+completely covers the body; ACCEPT it\n'
    '<answer>{"verdict": "add", "click": {"x": <f>, "y": <f>, "label": 0 or 1}}'
    '</answer>  -- none is complete: add a click (1=include missed part; '
    '0=exclude a specific cyan spill, touching neighbour, or background region)\n'
    '<answer>{"verdict": "reject"}</answer>'
    '  -- all candidates are hopeless/wrong object; DISCARD clicks & restart\n'
    '<answer>{"verdict": "abandon"}</answer>'
    '  -- there is NO real creature here; produce no mask\n'
    'x,y NORMALIZED [0,1] within ONE panel (the panels are identical crops). '
    'Use label 0 whenever cyan crosses the intended target boundary; put it '
    'inside the unwanted cyan region and never on another desired part of the target. '
    'GOOD IS A STRICT, NON-COMPARATIVE VERDICT: do not choose the least-bad '
    'candidate. Fine enclosed spaces between branches MAY be filled as part of '
    "one object's silhouette. What is forbidden is WHITE extending outside that "
    'object onto a visually separable neighbouring coral/plant, substrate lobe, '
    'or disconnected spill. Put label 0 on a WHITE pixel in that unwanted '
    'external/neighbor region. A negative click excludes that location from this '
    'target mask; it does not claim the neighbouring life is non-biological.'
)

DUPLICATE_FEEDBACK = (
    "Your previous requested click duplicated an existing "
    "click and made no change. Choose a genuinely different "
    "useful point, or answer good/reject/abandon."
)

MASK_VERIFY_FMT = (
    'Output: brief reasoning then EXACTLY ONE trailing tag. ALWAYS include '
    '"confidence" -- your probability (0.0-1.0) that this is a HIGH-QUALITY, tight '
    'segmentation of the described real life form (1.0 = near-perfect organism '
    'mask, 0.0 = wrong/background-filled mask):\n'
    '<answer>{"keep": true, "confidence": <0.0-1.0>, '
    '"complete_identity": true, "single_identity": true}</answer>   -- WHITE '
    'follows the complete intended life form closely, excludes background/'
    'neighbours, and contains exactly one identity/depth layer\n'
    '<answer>{"keep": false, "confidence": <0.0-1.0>, '
    '"complete_identity": <true|false>, "single_identity": <true|false>, '
    '"failure": "fragment|merge|wrong|background", '
    '"repair_click": {"x": <float>, "y": <float>, "label": 0 or 1}}</answer>  '
    '-- WHITE is only a '
    'piece of a larger identity, merges identities/depth layers, targets the '
    'wrong object, or is mostly non-biological background. Include repair_click '
    'only when one unambiguous click can improve this SAME target: label 1 on a '
    'missed continuation for fragment; label 0 inside wrongly included pixels '
    'for merge/background. Omit repair_click for wrong/unsupported targets. '
    'Repair x,y are NORMALIZED [0,1] in the FULL FRAME.'
)


def build_judge_prompt(
    description: str,
    *,
    duplicate_feedback: str = "",
    budget_reached: bool = False,
) -> str:
    """The candidate-selection prompt, shown with [full frame, candidate sheet]."""
    desc = description
    return (
        f"The FIRST image is the raw full frame. The discovery description "
        f"'{desc}' is an UNTRUSTED HYPOTHESIS, not proof that the proposed "
        f"pixels form one complete object. The SECOND review sheet has two "
        f"aligned rows for three SAM3 masks "
        f"#0,#1,#2 for '{desc}'. TOP = cyan overlay on identical crops "
        f"(green dot = foreground/include, red X = background/exclude). "
        f"BOTTOM = exact binary truth: WHITE pixels are "
        f"inside the mask and BLACK pixels are outside. Judge the WHITE "
        f"pixels, not biological structure still visible underneath the "
        f"translucent cyan. Fine enclosed spaces between branches may be "
        f"filled; judge the complete outer silhouette of the individual "
        f"object. The "
        f"accepted mask must cover "
        f"the complete visible target while "
        f"excluding substrate and visually separable neighbouring life. "
        f"One accepted mask must be exactly ONE complete biological "
        f"identity. A small branch/patch of a larger continuous branching "
        f"system is incomplete: add label 1 on the missed same-object "
        f"extent. A union of separate foreground/background colonies or "
        f"different branching systems is merged: add label 0 inside the "
        f"unwanted identity. Do this even when both identities are living "
        f"or share the same taxon. "
        f"On-screen logos, lettering, timestamps, grid marks, and other "
        f"video/UI overlays are never part of a biological mask. Reject "
        f"them or add a label-0 click on an included overlay spill. "
        f"Do not accept a broad merged region merely because it contains "
        f"the target. Never accept the closest candidate if it includes "
        f"pieces of a visually separable neighbouring coral/plant or a "
        f"substrate spill. If every candidate "
        f"spills into one specific unwanted "
        f"region, add a label-0 click inside that spill; if the target itself "
        f"is incomplete, add label 1 on the missed part. "
        f"{duplicate_feedback}"
        + (
            "NO CLICK BUDGET REMAINS: answer good only for a genuinely "
            "acceptable candidate; otherwise answer reject. Do not ask "
            "for another click.\n\n"
            if budget_reached else "\n\n"
        )
        + MM_JUDGE_FMT
    )


def geometry_fact(mask: np.ndarray, width: int, height: int) -> str:
    """Deterministic bbox and frame-edge facts, computed from the mask itself.

    These override any visual guess the verifier might make about location or
    frame clipping -- the crop can lie about both, the pixels cannot.
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return (
            "DETERMINISTIC BINARY-MASK GEOMETRY: the mask is EMPTY; "
            "touches frame edge(s)=none. "
        )
    mask_y, mask_x = np.where(mask)
    bbox_left, bbox_right = int(mask_x.min()), int(mask_x.max())
    bbox_top, bbox_bottom = int(mask_y.min()), int(mask_y.max())
    touched = [
        edge
        for edge, hit in (
            ("left", bbox_left == 0),
            ("right", bbox_right == width - 1),
            ("top", bbox_top == 0),
            ("bottom", bbox_bottom == height - 1),
        )
        if hit
    ]
    edge_fact = ",".join(touched) if touched else "none"
    return (
        "DETERMINISTIC BINARY-MASK GEOMETRY: normalized bbox "
        f"x={bbox_left / max(1, width - 1):.3f}.."
        f"{bbox_right / max(1, width - 1):.3f}, "
        f"y={bbox_top / max(1, height - 1):.3f}.."
        f"{bbox_bottom / max(1, height - 1):.3f}; "
        f"touches frame edge(s)={edge_fact}. These facts are computed from "
        "WHITE pixels and override any visual guess about location or frame "
        "clipping. "
    )


def repair_fact(repair_history: list[dict[str, Any]] | None) -> str:
    """The no-progress rule, shown once a proposal has been repaired before."""
    if not repair_history:
        return ""
    repair_summary = ", ".join(
        f"round {item.get('round')}: {item.get('failure', 'unknown')} "
        f"at ({float((item.get('click') or {}).get('x', 0.0)):.3f}, "
        f"{float((item.get('click') or {}).get('y', 0.0)):.3f}, "
        f"label={int((item.get('click') or {}).get('label', -1))})"
        for item in repair_history
    )
    return (
        "REPAIR HISTORY FOR THIS SAME PROPOSAL: " + repair_summary + ". "
        "If the latest mask is still invalid and SAM3 is cycling between "
        "fragment and merge failures, or no genuinely new actionable "
        "correction remains, reject it WITHOUT repair_click so the system "
        "can abandon this proposal and continue searching. Do not repeat "
        "or slightly move an earlier click merely to keep trying. This is "
        "a no-progress decision, not permission to accept a bad mask. "
    )


OCCLUSION_ADDENDUM = (
    "OCCLUSION AND VISIBILITY: an organism may be partially hidden behind "
    "another organism or object, or its far extent may fade below visibility "
    "in dark water. If the un-masked continuation is NOT clearly visible in "
    "the raw frame, treat the visible extent as the complete maskable target: "
    "a termination at an occluder, or at the point where the structure becomes "
    "indistinguishable from the background, IS a valid boundary, and "
    "complete_identity should be judged true. Reject as an incomplete fragment "
    "only when the SAME structure VISIBLY continues outside WHITE. "
)
"""Policy addendum, 2026-08-20: mask what is visible.

Kept OUT of the base prompt so the base stays byte-equal to the tuned
original; appended only when the caller opts in (`occlusion_addendum=True`,
wired from `ClickEngineConfig.keep_partial_fragments`).
"""


def build_verify_prompt(
    description: str,
    mask: np.ndarray,
    width: int,
    height: int,
    *,
    repair_history: list[dict[str, Any]] | None = None,
    allow_all_life: bool = True,
    occlusion_addendum: bool = False,
) -> str:
    """The mask-verification rubric, shown with [raw frame, context, review sheet]."""
    desc = description or "the masked region"
    if allow_all_life:
        subject = (
            "real, distinct marine life: an animal, coherent animal colony, "
            "sponge, coral, macroalga, seagrass, or other living organism"
        )
    else:
        subject = "a real, distinct living animal"
    return (
        "The FIRST image is the raw full frame and is the coordinate reference. "
        "The SECOND image is the same full frame with the candidate in cyan. "
        "The THIRD review sheet has two aligned rows for one zoomed candidate "
        f"mask for '{desc}'. TOP = cyan overlay. BOTTOM = exact binary truth: WHITE pixels "
        "are inside the mask and BLACK pixels are outside. "
        + geometry_fact(mask, width, height)
        + repair_fact(repair_history)
        + (OCCLUSION_ADDENDUM if occlusion_addendum else "")
        + "Cyan/WHITE having a smooth, coherent silhouette is NOT evidence "
        "that an organism exists: SAM3 can create convincing blobs from "
        "water, shadow, haze, or substrate. The untouched FIRST image must "
        "show biological texture, branching, anatomy, or a credible natural "
        "boundary at the exact mask location. If the hypothesis says the "
        "target is frame-clipped but touches frame edge(s)=none, reject it "
        "as wrong/background. "
        f"Decide whether WHITE tightly segments {subject}. The discovery description is an "
        "UNTRUSTED HYPOTHESIS; infer the true biological identity and its full "
        "extent from the images. Judge mask membership, not life "
        "still visible beneath translucent cyan. For branching organisms, "
        "fine enclosed inter-branch spaces may be WHITE as part of the outer "
        "silhouette. Reject masks that are mostly background, extend well "
        "outside the target silhouette, merge visually separable neighbours "
        "or foreground/background depth layers, "
        "or select the wrong object, even if some real "
        "life is visible in the crop. Reject a small branch, appendage, or "
        "patch when the same organism/colony visibly continues outside WHITE; "
        "that is an incomplete fragment, not a valid separate target. Reject "
        "a union of two distinct branching systems even when both are life or "
        "the same taxon. Trace the WHITE boundary around the whole candidate: "
        "every non-frame-edge termination must coincide with a true visible "
        "outer boundary or a real occluder. If WHITE cuts through visually "
        "continuous branches of the same colony, complete_identity must be "
        "false. If a path through WHITE crosses an occlusion/depth transition "
        "or joins distinct branch systems, single_identity must be false. "
        "Set keep=true only when BOTH identity fields are true. On-screen "
        "logos, lettering, timestamps, "
        "grid marks, and other video/UI overlays are non-biological; reject "
        "a candidate that masks them instead of the described organism. Any "
        "green dot is an include click and "
        "must lie on the intended target; any red X is an exclude click and "
        "must lie outside it. "
        "\n\n" + MASK_VERIFY_FMT
    )
