"""SAM3 click-mode service.

Wraps the `processor.set_image` / `model.predict_inst` pair that the fork's
image model exposes. Everything above this layer talks to `Sam3Backend`, so the
click engine can be exercised against a fake backend without CUDA.

Ported from `Sam3PointService` in `nibi_model_compare/som_missed_creatures.py`
and `_sam3_raw` in `scripts/click_engine_probe.py`. The mask-selection policy
that both open-coded now comes from `geometry.select_in_band` -- the original
comment in click_engine_probe even said it was "mirroring
Sam3PointService.group_segment".
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from ..geometry import select_in_band


@runtime_checkable
class Sam3Backend(Protocol):
    """The slice of SAM3 this pipeline actually uses."""

    def set_image(self, image: Any) -> Any:
        """Embed one image; returns an inference state reused across clicks."""

    def predict_inst(
        self,
        state: Any,
        *,
        point_coords: np.ndarray | None = ...,
        point_labels: np.ndarray | None = ...,
        box: np.ndarray | None = ...,
        multimask_output: bool = ...,
    ) -> tuple[Any, Any, Any]:
        """Return `(masks, scores, logits)` for the given prompts."""


def _log(message: str) -> None:
    print(message)


def as_pil_rgb(image: Any) -> Any:
    """Coerce an image to PIL RGB, which is what the SAM3 processor expects.

    The rest of this pipeline works in OpenCV's BGR numpy convention, so the
    conversion happens once, here. Handing the processor a BGR array instead
    silently produces masks of the wrong SHAPE as well as the wrong colour --
    a centre click returned (3, 1280, 3) rather than (3, 720, 1280) -- which
    then flows downstream as a plausible-looking but meaningless mask.
    """
    from PIL import Image  # noqa: PLC0415

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)


def _to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _normalise_prediction(masks: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a possibly-batched (1, N, H, W) prediction to (N, H, W)."""
    masks_np = _to_numpy(masks)
    scores_np = _to_numpy(scores)
    if masks_np.ndim == 4:
        masks_np = masks_np[0]
        scores_np = scores_np[0] if scores_np.ndim >= 1 else scores_np
    return masks_np.astype(bool), np.atleast_1d(scores_np)


def _empty_result(group: dict[str, Any], height: int, width: int, status: str,
                  clicks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "creature_id": int(group.get("id", -1)),
        "description": str(group.get("description", "")),
        "mask": np.zeros((height, width), dtype=bool),
        "score": 0.0,
        "area_px": 0,
        "select_reason": status,
        "spatial_match": status,
        "clicks_used": clicks,
    }


def valid_clicks(clicks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only well-formed clicks. The parser should already have done this."""
    return [
        c
        for c in clicks
        if isinstance(c.get("x"), (int, float))
        and isinstance(c.get("y"), (int, float))
        and c.get("label") in (0, 1)
    ]


class Sam3PointService:
    """Run SAM3 click mode against an image, one embedding per image.

    Clicks arrive normalised to the full frame; SAM3 wants pixels, so they are
    scaled here and nowhere else.
    """

    def __init__(self, model: Sam3Backend, processor: Any = None):
        self.model = model
        self.processor = processor if processor is not None else model
        if not hasattr(model, "predict_inst"):
            raise RuntimeError(
                "SAM3 image model does not expose predict_inst; rebuild with "
                "enable_inst_interactivity=True."
            )

    # -- low level ---------------------------------------------------------

    def raw_predict(
        self,
        image: Any,
        *,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        multimask: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """All candidate masks and scores for one prompt. Coordinates in pixels."""
        state = self.processor.set_image(as_pil_rgb(image))
        kwargs: dict[str, Any] = {"multimask_output": multimask}
        if point_coords is not None:
            kwargs["point_coords"] = np.asarray(point_coords, dtype=np.float32)
            kwargs["point_labels"] = np.asarray(point_labels, dtype=np.int64)
        if box is not None:
            kwargs["box"] = np.asarray(box, dtype=np.float32)
        masks, scores, _ = self.model.predict_inst(state, **kwargs)
        return _normalise_prediction(masks, scores)

    # -- group level -------------------------------------------------------

    def group_segment(
        self, image: Any, groups: list[dict[str, Any]], size_hw: tuple[int, int] | None = None
    ) -> list[dict[str, Any]]:
        """Segment each creature group, using all of its clicks jointly.

        Multiple positive clicks let SAM3 disambiguate elongated organisms;
        negative clicks exclude substrate or a neighbouring animal. One failed
        group yields an empty result rather than aborting the frame.
        """
        state = self.processor.set_image(as_pil_rgb(image))
        height, width = size_hw if size_hw is not None else _image_size(image)

        results: list[dict[str, Any]] = []
        for group in groups:
            good = valid_clicks(group.get("clicks") or [])
            if not good:
                results.append(_empty_result(group, height, width, "click_mode_empty", []))
                results[-1]["select_reason"] = "no_clicks"
                continue

            coords = np.array(
                [[float(c["x"]) * width, float(c["y"]) * height] for c in good], dtype=np.float32
            )
            labels = np.array([int(c["label"]) for c in good], dtype=np.int64)

            try:
                masks, scores, _ = self.model.predict_inst(
                    state, point_coords=coords, point_labels=labels, multimask_output=True
                )
            except Exception as exc:  # noqa: BLE001 - one bad group must not kill the frame
                _log(
                    f"[sam3] group_segment failed for creature {group.get('id')!r} "
                    f"({group.get('description')!r}): {type(exc).__name__}: {exc}"
                )
                results.append(_empty_result(group, height, width, "click_mode_error", good))
                continue

            masks_np, scores_np = _normalise_prediction(masks, scores)
            if masks_np.size == 0:
                results.append(_empty_result(group, height, width, "click_mode_empty", good))
                continue

            index, reason = select_in_band(masks_np, scores_np)
            mask = masks_np[index]
            results.append(
                {
                    "creature_id": int(group.get("id", -1)),
                    "description": str(group.get("description", "")),
                    "mask": mask,
                    "score": float(scores_np[index]),
                    "area_px": int(mask.sum()),
                    "select_reason": reason,
                    "spatial_match": "click_mode",
                    "clicks_used": good,
                }
            )
        return results


def _image_size(image: Any) -> tuple[int, int]:
    """(height, width) for a PIL image or an HxWx3 array.

    PIL exposes `.size` as (width, height); numpy exposes `.size` as an element
    count, so the tuple check matters.
    """
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:  # PIL
        width, height = size
        return int(height), int(width)
    arr = np.asarray(image)
    return int(arr.shape[0]), int(arr.shape[1])


def find_bpe_path() -> str | None:
    """Locate the BPE vocab, preferring SAM3_BPE_PATH then the packaged asset."""
    import os  # noqa: PLC0415
    from importlib import resources  # noqa: PLC0415

    env_path = os.environ.get("SAM3_BPE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    try:
        packaged = resources.files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz")
        if packaged.is_file():
            return str(packaged)
    except (ModuleNotFoundError, AttributeError):
        pass
    return None  # build_sam3_image_model resolves its own default


def build_sam3_service(
    device: str = "cuda",
    bpe_path: str | None = None,
    *,
    confidence_threshold: float = 0.5,
) -> Sam3PointService:
    """Build the real SAM3 click-mode service. Requires the [sam3] extra.

    `enable_inst_interactivity=True` is what exposes `predict_inst`; without it
    the model cannot take point prompts at all.
    """
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
    from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415

    model = build_sam3_image_model(
        bpe_path=bpe_path if bpe_path is not None else find_bpe_path(),
        device=device,
        enable_inst_interactivity=True,
    )
    processor = Sam3Processor(
        model, device=device, confidence_threshold=confidence_threshold
    )
    return Sam3PointService(model, processor)
