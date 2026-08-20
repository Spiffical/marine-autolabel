"""Image plumbing for multimodal requests.

Reimplemented rather than imported from the SAM3 fork: these are generic
utilities, and keeping them here means the LLM layer has no `sam3` dependency,
so it installs and tests without CUDA.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL import Image

_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def image_to_base64(path: str, max_edge: int | None = None) -> tuple[str | None, str | None]:
    """Return `(base64_data, mime_type)` for an image, optionally downscaled.

    When `max_edge` is set and the image's longest side exceeds it, the image is
    re-encoded as JPEG at that bound -- this is the lever the retry loop pulls
    when a request comes back too large. Returns `(None, None)` on any failure,
    matching the original behaviour: a missing frame should skip a block, not
    kill the run.
    """
    mime_type = _MIME_BY_EXT.get(os.path.splitext(path)[1].lower(), "image/jpeg")
    try:
        if max_edge and max_edge > 0:
            with Image.open(path) as img:
                img = img.convert("RGB")
                width, height = img.size
                if max(width, height) > max_edge:
                    scale = max_edge / float(max(width, height))
                    img = img.resize(
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90, optimize=True)
                    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"

        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8"), mime_type
    except Exception as exc:  # noqa: BLE001 - a bad frame must not abort the run
        print(f"Error converting image to base64: {exc}")
        return None, None


def cap_images(messages: list[dict[str, Any]], max_images: int | None) -> list[dict[str, Any]]:
    """Keep at most `max_images` image blocks across all user messages.

    The most recent images are the ones kept -- in this pipeline the newest frame
    and its overlays carry the information, while older turns are context. Text
    blocks are always preserved.
    """
    if max_images is None or max_images <= 0:
        return messages

    positions: list[tuple[int, int]] = []
    for m_idx, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for c_idx, item in enumerate(content):
            if isinstance(item, dict) and item.get("type") in {"image", "image_url"}:
                positions.append((m_idx, c_idx))

    if len(positions) <= max_images:
        return messages

    keep = set(positions[-max_images:])
    trimmed: list[dict[str, Any]] = []
    for m_idx, message in enumerate(messages):
        copy = dict(message)
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, list):
            copy["content"] = [
                item
                for c_idx, item in enumerate(content)
                if not (
                    isinstance(item, dict)
                    and item.get("type") in {"image", "image_url"}
                    and (m_idx, c_idx) not in keep
                )
            ]
        trimmed.append(copy)
    return trimmed
