"""FathomNet baseline detectors.

Ported from `scripts/run_fathomnet_baselines.py`. These are the published
comparison points: supervised detectors trained on FathomNet imagery, run on the
same fixed frames as the pipeline.

Deliberately API-FREE. Weights come from the Hugging Face cache and inference is
local, so a baseline sweep costs nothing against the Anthropic key. That is the
point of having them -- they can be re-run freely whenever the frame set changes.

A caveat that governs any comparison drawn from these: a detector emits BOXES
over a closed class vocabulary, while the pipeline emits instance MASKS over
open-vocabulary "any visible organism". Counts are not directly comparable, and
a detector scoring zero on a frame may simply mean nothing in the frame belongs
to its classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODELS: dict[str, dict[str, str]] = {
    "fathomnet_mbari_315k_yolov8": {
        "display_name": "FathomNet MBARI 315k YOLOv8",
        "repo_id": "FathomNet/MBARI-315k-yolov8",
        "filename": "mbari_315k_yolov8.pt",
    },
    "fathomnet_megalodon_2023_yolov8": {
        "display_name": "FathomNet Megalodon 2023 YOLOv8",
        "repo_id": "FathomNet/megalodon-2023-yolov8",
        "filename": "mbari-megalodon-yolov8x.pt",
    },
    "fathomnet_benthic_2025": {
        "display_name": "FathomNet Benthic 2025",
        "repo_id": "FathomNet/2025-MBARI-Benthic-Supercategory-Object-Detector",
        "filename": "best.pt",
    },
    "fathomnet_midwater_2025": {
        "display_name": "FathomNet Midwater 2025",
        "repo_id": "FathomNet/2025-MBARI-Midwater-Supercategory-Object-Detector",
        "filename": "best.pt",
    },
}


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "xyxy": [round(float(v), 3) for v in self.xyxy],
            "confidence": round(float(self.confidence), 6),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
        }


def extract_detections(result: Any) -> list[Detection]:
    """Pull detections out of an Ultralytics result object.

    An empty `boxes` is a legitimate outcome -- the frame contains nothing from
    this detector's classes -- and yields an empty list rather than an error.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    def to_list(value: Any) -> list:
        return value.detach().cpu().tolist() if hasattr(value, "detach") else list(value)

    names = result.names
    detections: list[Detection] = []
    # strict=True: the three arrays are parallel by construction, so a length
    # mismatch is a malformed result. Silently truncating would drop detections.
    for box, score, class_index in zip(
        to_list(boxes.xyxy), to_list(boxes.conf), to_list(boxes.cls), strict=True
    ):
        class_id = int(class_index)
        detections.append(
            Detection(
                xyxy=tuple(float(v) for v in box),
                confidence=float(score),
                class_id=class_id,
                class_name=str(names[class_id]),
            )
        )
    return detections


def filter_by_confidence(detections: list[Detection], threshold: float) -> list[Detection]:
    """Keep detections at or above `threshold`, highest confidence first."""
    kept = [d for d in detections if d.confidence >= threshold]
    return sorted(kept, key=lambda d: d.confidence, reverse=True)


def threshold_slug(value: float) -> str:
    """A filesystem-safe tag for a confidence threshold: 0.25 -> `conf025`."""
    return "conf" + f"{float(value):.2f}".replace("0.", "").replace(".", "")


def summarize(detections: list[Detection], threshold: float) -> dict[str, Any]:
    """Counts per class at one threshold, for the comparison table."""
    kept = filter_by_confidence(detections, threshold)
    per_class: dict[str, int] = {}
    for detection in kept:
        per_class[detection.class_name] = per_class.get(detection.class_name, 0) + 1
    return {
        "threshold": threshold,
        "n_detections": len(kept),
        "n_classes": len(per_class),
        "per_class": dict(sorted(per_class.items())),
        "max_confidence": max((d.confidence for d in kept), default=0.0),
    }
