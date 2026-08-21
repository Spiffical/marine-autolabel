"""Building the discovery request: views and the prompt that describes them.

Ported from `run_presentation_custom_flow._mask_guided_discovery_prompt` and the
content assembly ~250 lines away in `_find_mask_guided_groups`.

Those two were maintained separately, and the prompt refers to images by
POSITION -- "The FOURTH image is an outline-only map". Adding the outline view
on 2026-08-18 therefore required editing the ordinal words and inserting the
image at the matching index, in two places. Getting that wrong does not raise;
it tells the model image four is a focus crop when it is a coverage map, and the
only symptom is worse clicks.

Here the view list is the single source of truth: ordinals are generated from
it, and `build_content` refuses a mismatched image set. The prompt wording is
otherwise byte-identical to the original -- these strings are tuned and are not
mine to improve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ORDINALS = ("first", "second", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH")


@dataclass(frozen=True)
class View:
    """One image shown to the discovery model, and the sentence describing it."""

    key: str
    template: str

    def describe(self, ordinal: str) -> str:
        return self.template.format(ordinal=ordinal)


RAW = View("raw", "The {ordinal} image is the untouched raw TARGET FRAME. ")
GRID = View(
    "grid",
    "The {ordinal} image is the same target with lightly translucent green "
    "regions showing life masks already accepted from earlier stages; its "
    "10x10 grid gives normalized coordinate context. ",
)
STRONG = View(
    "strong",
    "The {ordinal} image is a strong green coverage map of the same accepted "
    "masks. Use the strong map to decide whether a particular structure is "
    "already covered, and use the raw and light views to inspect its biological "
    "boundaries. Compare the raw and masked target pixel-for-pixel: faint life "
    "beside or behind a green region is still missed when its own structure is "
    "not green. Do not assume a cluster is fully covered merely because nearby "
    "organisms are green. ",
)
OUTLINE = View(
    "outline",
    "The {ordinal} image is an outline-only map of accepted masks: green "
    "contours show their outer boundaries while leaving the raw interior "
    "visible. This is the decisive view for crowded branching scenes. A "
    "separate coral, whip, or colony visible through the filled inter-branch "
    "space of an accepted outer silhouette is NOT thereby segmented; report it "
    "if it has its own visually separable branching system or depth layer. ",
)
CLAHE = View(
    "clahe",
    "The {ordinal} image is a LOCAL-CONTRAST-ENHANCED copy of the target "
    "frame. Organisms whose texture blends into the lit seafloor are far "
    "easier to see here than in the raw frame. Use it to FIND such organisms, "
    "but take all appearance and boundary judgements from the raw frame; the "
    "enhancement exaggerates noise as well as structure. ",
)
"""Added 2026-08-21 after a coverage audit: a four-quadrant sweep accepted only
organisms silhouetted against dark water and missed nearly everything textured
against the lit seafloor. Global views carry too little local contrast there."""

FOCUS_RAW = View(
    "focus_raw",
    "The {ordinal} image is an enlarged untouched crop of the required focus region. ",
)
FOCUS_STRONG = View(
    "focus_strong",
    "The {ordinal} image is the matching enlarged strong-mask crop, retaining "
    "the full-frame coordinate grid. All returned coordinates must remain "
    "normalized to the FULL FRAME, not the crop. ",
)

BASE_VIEWS = (RAW, GRID, STRONG, OUTLINE)
FOCUS_VIEWS = (FOCUS_RAW, FOCUS_STRONG)

TARGET_CONTRACT = (
    "Find every real, visually detectable marine organism or coherent colony in "
    "the TARGET FRAME that is NOT adequately covered by green. Targets include "
    "mobile animals; corals, sea fans, sea whips, sponges, anemones, hydroids "
    "and other sessile or colonial animals; and coherent macroalgae, seagrass, "
    "or other plant-like marine life. Treat each visually separable organism or "
    "coherent colony as a target. Include small, camouflaged, stationary, and "
    "frame-clipped life when credible biological structure or temporal support "
    "is visible. Do not click green-covered life, marine snow, bare substrate, "
    "debris, shell fragments, shadows, unsupported blobs, or any on-screen logo, "
    "text, timestamp, grid line, or other video/UI overlay. If an organism "
    "passes behind an overlay, click only a visible biological part outside it. "
    "Return an empty list only after an exhaustive scan finds no supported "
    "missed life.\n\n"
    "Return one small click GROUP per missed target. Every group must start "
    "with label 1 on a safe interior part of that target. Label 0 means EXCLUDE "
    "THIS LOCATION FROM THIS TARGET'S MASK; it does not mean that the location "
    "contains no life. Add one or two label-0 clicks when the target touches "
    "substrate, a green mask, or a visually separable neighbour that SAM3 may "
    "merge. Put each negative inside the unwanted region, never on another "
    "desired part of the same organism. High-value negative locations include a "
    "touching but visually separable coral or plant, a substrate lobe protruding "
    "beyond the target's outer silhouette, or a disconnected spill. Do not use a "
    "negative click solely to carve small spaces between fine branches: filling "
    "those enclosed spaces is acceptable when the mask still follows one "
    "complete object's silhouette. ONE GROUP MUST REPRESENT EXACTLY ONE COMPLETE "
    "BIOLOGICAL IDENTITY. Never redefine a branch, appendage, small visible "
    "patch, or other subregion of a larger continuous organism/colony as a "
    "separate target; place positives across the complete visible extent "
    "instead. Conversely, never put positives from visually separable foreground "
    "and background organisms/colonies into one group, even if they overlap, "
    "touch, look similar, or share a taxon. A depth-layer change, occlusion "
    "boundary, or separate branching system requires separate groups. Use a "
    "second label-1 click only for another part of the same elongated or "
    "fragmented target. Otherwise, one central positive click is best.\n\n"
    "Coordinate accuracy is critical. Before submitting, verify that every "
    "positive coordinate visibly lands ON the described target in the grid, not "
    "in nearby water or between branches. For branching life, prefer a thick, "
    "high-contrast branch or solid base that is safely inside the target.\n\n"
    "Output free-text reasoning followed by EXACTLY ONE trailing tag:\n"
    '<answer>{"missed_creatures":[{"id":1,"description":"<short>",'
    '"clicks":[{"x":<float>,"y":<float>,"label":1},'
    '{"x":<float>,"y":<float>,"label":0}]}]}</answer>'
)


def discovery_views(
    *, has_focus_crops: bool, has_clahe: bool = False
) -> tuple[View, ...]:
    """The ordered views for one discovery request."""
    views = BASE_VIEWS + ((CLAHE,) if has_clahe else ())
    return views + (FOCUS_VIEWS if has_focus_crops else ())


def build_discovery_prompt(
    *, pass_instruction: str, has_focus_crops: bool = False, has_clahe: bool = False
) -> str:
    """The recall-first all-life discovery contract.

    Ordinals are derived from the view list, so a view cannot be added without
    the description renumbering itself.
    """
    views = discovery_views(has_focus_crops=has_focus_crops, has_clahe=has_clahe)
    if len(views) > len(ORDINALS):
        raise ValueError(f"no ordinal for view {len(views)}; extend ORDINALS")

    described = "".join(view.describe(ORDINALS[index]) for index, view in enumerate(views))
    return (
        described
        + "Subsequent images are raw REFERENCE FRAMES from nearby times. "
        + pass_instruction
        + TARGET_CONTRACT
    )


def build_content(
    *,
    view_paths: dict[str, str],
    neighbour_paths: list[str],
    pass_instruction: str,
) -> list[dict[str, Any]]:
    """Assemble the user content, images in exactly the order described.

    `view_paths` must supply every view and nothing else -- a missing or unknown
    key means the prompt and the images have drifted apart, which is silent
    corruption rather than a crash, so it is refused here.
    """
    has_focus_crops = FOCUS_RAW.key in view_paths or FOCUS_STRONG.key in view_paths
    has_clahe = CLAHE.key in view_paths
    views = discovery_views(has_focus_crops=has_focus_crops, has_clahe=has_clahe)
    expected = [view.key for view in views]

    missing = [key for key in expected if key not in view_paths]
    unknown = [key for key in view_paths if key not in expected]
    if missing or unknown:
        raise ValueError(
            f"discovery views do not match the prompt: missing={missing}, "
            f"unexpected={unknown}, expected order={expected}"
        )

    content: list[dict[str, Any]] = [
        {"type": "image", "image": view_paths[key]} for key in expected
    ]
    content.extend({"type": "image", "image": path} for path in neighbour_paths)
    content.append(
        {
            "type": "text",
            "text": build_discovery_prompt(
                pass_instruction=pass_instruction,
                has_focus_crops=has_focus_crops,
                has_clahe=has_clahe,
            ),
        }
    )
    return content
