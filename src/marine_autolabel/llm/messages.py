"""Convert the agent's OpenAI-style message list into Anthropic content blocks.

Kept separate from the transport in `claude.py` so the conversion is testable
without the anthropic SDK or a network. It is pure.

The agent emits messages in a loose shape that has accumulated over time:
system content may be a string or a block list; user content may be a bare
string, a list of `{"type": "text"}` / `{"type": "image", "image": path}`
blocks, or OpenAI `image_url` blocks carrying data URIs. All of it is
normalised here.
"""
from __future__ import annotations

from typing import Any

from .images import cap_images, image_to_base64


def flatten_system_content(content: Any) -> str:
    """System content is a plain string today, but be lenient about block lists."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    return str(content or "")


def normalize_content(content: Any) -> list[Any]:
    """Coerce user/assistant content into a list of blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def convert_block(block: Any, *, image_max_edge: int | None) -> dict[str, Any] | None:
    """Convert one block to Anthropic shape, or None if it carries nothing."""
    if not isinstance(block, dict):
        if isinstance(block, str) and block.strip():
            return {"type": "text", "text": block}
        return None

    btype = str(block.get("type", "")).lower()

    if btype in {"text", "output_text"}:
        text = block.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        return {"type": "text", "text": text}

    if btype == "image":
        image_path = block.get("image")
        if not isinstance(image_path, str) or not image_path:
            return None
        # '?' in a path would be read as a query separator downstream.
        data, mime = image_to_base64(image_path.replace("?", "%3F"), max_edge=image_max_edge)
        if not data:
            return None
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}

    if btype == "image_url":
        url_field = block.get("image_url")
        url = url_field.get("url") if isinstance(url_field, dict) else url_field
        if not isinstance(url, str) or not url.startswith("data:"):
            return None  # remote URLs are not expected from this codepath
        try:
            header, payload = url.split(",", 1)
        except ValueError:
            return None
        mime = header.split(";")[0][len("data:"):] or "image/jpeg"
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": payload}}

    return None


def to_anthropic(
    messages: list[dict[str, Any]],
    *,
    image_max_edge: int | None,
    max_images: int | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Split out the system prompt and convert the rest to Anthropic messages.

    Returns `(system_text, messages)`. The image cap is applied *before* base64
    inflation, so dropped images are never encoded.
    """
    system_chunks: list[str] = []
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            piece = flatten_system_content(msg.get("content"))
            if piece.strip():
                system_chunks.append(piece)
            continue
        if role not in {"user", "assistant"}:
            continue
        normalized.append({"role": role, "content": normalize_content(msg.get("content"))})

    converted: list[dict[str, Any]] = []
    for msg in cap_images(normalized, max_images):
        blocks = [
            block
            for block in (
                convert_block(part, image_max_edge=image_max_edge) for part in msg["content"]
            )
            if block is not None
        ]
        if blocks:
            converted.append({"role": msg["role"], "content": blocks})

    return "\n\n".join(system_chunks).strip(), converted
