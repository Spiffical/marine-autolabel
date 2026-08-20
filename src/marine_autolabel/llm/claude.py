"""Anthropic transport with the pipeline's retry and downscale policy.

Ported from `sam3/agent/client_claude.py`. The message conversion moved to
`messages.py`; what remains here is the request loop and its recovery rules:

  rate limit / 5xx / APIError -> exponential backoff, capped at 30s
  request too large           -> shrink images 20% at a time to a floor, then
                                 fall back to a single image per request
  no text block in response   -> short backoff and retry, since an empty
                                 response is usually transient
  truncated by max_tokens     -> retry with a doubled budget. A response cut
                                 off mid-tag parses as "nothing found" by the
                                 lenient parsers, so silently accepting it
                                 would show up as poor recall, not an error

Returning None rather than raising is deliberate and load-bearing: the frame
driver treats a None as an API failure, declines to cache that frame, and
retries it on the next run. That is what makes long fan-outs resumable.
"""
from __future__ import annotations

import os
import time
from typing import Any

import anthropic

from .messages import to_anthropic

ALLOWED_EFFORTS = {"low", "medium", "high", "xhigh", "max"}

MAX_COMPLETION_TOKENS = 32768
"""Ceiling for the truncation-retry ladder."""

DEFAULT_IMAGE_MAX_EDGE = 1024
DEFAULT_IMAGE_MIN_EDGE = 384
DEFAULT_TIMEOUT_SECONDS = 120.0


def _log(message: str) -> None:
    """Single seam for output. Still prints, so run logs are unchanged."""
    print(message)


def _env_int(*names: str, default: int) -> int:
    """First of `names` that parses as an int. Later names are legacy aliases."""
    for name in names:
        raw = os.environ.get(name)
        if raw:
            try:
                return int(raw)
            except ValueError:
                continue
    return default


def _next_smaller_image_edge(current_edge: int | None, floor_edge: int) -> int | None:
    if current_edge is None or current_edge <= floor_edge:
        return None
    reduced = max(floor_edge, int(current_edge * 0.8))
    return None if reduced >= current_edge else reduced


def extract_text(response: Any) -> str | None:
    """Join the text blocks of a response, or None if it carries none."""
    blocks = getattr(response, "content", None)
    if not blocks:
        return None
    out: list[str] = []
    for block in blocks:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
        else:
            continue
        if isinstance(text, str) and text.strip():
            out.append(text)
    return "\n".join(out).strip() if out else None


def send_claude_request(
    messages: list[dict[str, Any]],
    *,
    model: str = "claude-sonnet-4-6",
    api_key: str | None = None,
    max_tokens: int = 1024,
    effort: str | None = None,
    server_url: str | None = None,  # accepted for signature symmetry; ignored
    enable_prompt_cache: bool = True,
    max_retries: int = 5,
) -> str | None:
    """Send a message list to Claude and return the assistant text, or None.

    The signature mirrors the OpenAI-compatible adapter so this can be bound with
    `functools.partial` and handed to the agent loop unchanged.
    """
    del server_url

    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        _log("Anthropic request failed: ANTHROPIC_API_KEY is not set.")
        return None

    resolved_effort = (effort or os.environ.get("ANTHROPIC_EFFORT", "")).strip().lower()
    if resolved_effort and resolved_effort not in ALLOWED_EFFORTS:
        _log(
            "Anthropic request failed: effort must be one of "
            f"{sorted(ALLOWED_EFFORTS)}, got {resolved_effort!r}."
        )
        return None

    try:
        timeout_seconds = float(
            os.environ.get("ANTHROPIC_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        )
    except ValueError:
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS

    client = anthropic.Anthropic(api_key=resolved_key, timeout=max(10.0, timeout_seconds))

    # MAL_* are the current names; SAM3_* are honoured so existing shell
    # environments and SLURM scripts keep working.
    image_max_edge = _env_int(
        "MAL_IMAGE_MAX_EDGE", "SAM3_AGENT_IMAGE_MAX_EDGE", default=DEFAULT_IMAGE_MAX_EDGE
    )
    image_min_edge = max(
        128,
        _env_int("MAL_IMAGE_MIN_EDGE", "SAM3_AGENT_IMAGE_MIN_EDGE", default=DEFAULT_IMAGE_MIN_EDGE),
    )
    forced_max_images: int | None = _env_int(
        "MAL_MAX_IMAGES_PER_REQUEST", "SAM3_MAX_IMAGES_PER_REQUEST", default=0
    ) or None

    current_edge: int | None = image_max_edge if image_max_edge > 0 else None
    budget = max(int(max_tokens), 1)

    for attempt in range(max_retries):
        system_text, anthropic_messages = to_anthropic(
            messages, image_max_edge=current_edge, max_images=forced_max_images
        )
        if not anthropic_messages:
            _log("Anthropic request skipped: no convertible user/assistant messages.")
            return None

        # Mark the end of the system prompt as an ephemeral cache breakpoint.
        # The agent's preamble is stable across frames, so this stops it being
        # re-tokenised on every call.
        system_payload: Any = system_text
        if enable_prompt_cache and system_text:
            system_payload = [
                {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}
            ]

        effort_suffix = f", effort={resolved_effort}" if resolved_effort else ""
        _log(
            f"[Claude] Calling model {model} "
            f"(attempt {attempt + 1}/{max_retries}{effort_suffix})..."
        )

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": budget,
                "messages": anthropic_messages,
            }
            if system_payload:
                kwargs["system"] = system_payload
            if resolved_effort:
                kwargs["output_config"] = {"effort": resolved_effort}
            response = client.messages.create(**kwargs)

        except anthropic.RateLimitError as exc:
            wait = min(30, 2**attempt)
            _log(f"[Claude] Rate limit: {exc}. Sleeping {wait}s and retrying...")
            time.sleep(wait)
            continue

        except anthropic.APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            error_text = str(exc)
            _log(f"[Claude] APIStatusError {status}: {error_text[:300]}")
            normalized = error_text.lower()

            too_large = (
                "context" in normalized
                or "prompt is too long" in normalized
                or "request_too_large" in normalized
                or ("image" in normalized and "too large" in normalized)
            )
            if too_large:
                next_edge = _next_smaller_image_edge(current_edge, image_min_edge)
                if next_edge is not None:
                    _log(f"[Claude] Retrying with image_max_edge={next_edge}")
                    current_edge = next_edge
                    continue
                if forced_max_images is None or forced_max_images > 1:
                    forced_max_images = 1
                    _log("[Claude] Retrying with at most 1 image per request.")
                    continue

            if status and 500 <= int(status) < 600:
                wait = min(30, 2**attempt)
                _log(f"[Claude] Server error {status}. Sleeping {wait}s and retrying...")
                time.sleep(wait)
                continue

            return None

        except anthropic.APIError as exc:
            wait = min(30, 2**attempt)
            _log(f"[Claude] APIError: {exc}. Sleeping {wait}s and retrying...")
            time.sleep(wait)
            continue

        except Exception as exc:  # noqa: BLE001 - never let one frame kill a fan-out
            _log(f"[Claude] Unexpected error: {type(exc).__name__}: {exc}")
            return None

        stop_reason = getattr(response, "stop_reason", None)
        text = extract_text(response)

        # A response cut off at the token limit may carry text that ends
        # mid-tag. The parsers would read that as an empty result, so treat it
        # as a failed call and buy more room rather than believing it.
        if stop_reason == "max_tokens" and attempt + 1 < max_retries:
            grown = min(budget * 2, MAX_COMPLETION_TOKENS)
            if grown > budget:
                _log(
                    f"[Claude] TRUNCATED at max_tokens={budget}; the answer tag may be "
                    f"incomplete. Retrying with max_tokens={grown}."
                )
                budget = grown
                continue
            _log(
                f"[Claude] TRUNCATED at max_tokens={budget}, already at the "
                f"{MAX_COMPLETION_TOKENS} ceiling; returning what was produced."
            )

        if text is not None:
            if stop_reason == "max_tokens":
                _log("[Claude] WARNING: returning text from a truncated response.")
            return text

        usage = getattr(response, "usage", None)
        _log(
            "[Claude] No text block in response "
            f"(stop_reason={stop_reason!r}, "
            f"output_tokens={getattr(usage, 'output_tokens', None)!r}, budget={budget})."
        )
        if attempt + 1 < max_retries:
            wait = min(5, 2**attempt)
            _log(f"[Claude] Empty text content in response. Sleeping {wait}s and retrying...")
            time.sleep(wait)
            continue
        _log("[Claude] Empty text content after final attempt; returning None.")
        return None

    _log("[Claude] Exhausted retries.")
    return None
