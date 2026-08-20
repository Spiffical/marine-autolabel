"""Transport policy: retries, image downscaling, and the None contract.

Returning None on failure is load-bearing -- the frame driver reads it as an API
failure, refuses to cache the frame, and retries on the next run. These tests pin
that behaviour down without touching the network.
"""
from __future__ import annotations

import types

import pytest

from marine_autolabel.llm import claude
from marine_autolabel.llm.claude import _env_int, _next_smaller_image_edge, extract_text


class FakeRateLimit(Exception):
    pass


class FakeStatusError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class FakeAPIError(Exception):
    pass


def _text_response(text):
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=types.SimpleNamespace(output_tokens=5),
    )


def _empty_response():
    return types.SimpleNamespace(content=[], stop_reason="max_tokens",
                                 usage=types.SimpleNamespace(output_tokens=0))


@pytest.fixture
def transport(monkeypatch):
    """Install a scripted fake Anthropic client; returns the call recorder."""
    calls: list[dict] = []
    script: list = []

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            outcome = script.pop(0) if script else _text_response("ok")
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setattr(
        claude,
        "anthropic",
        types.SimpleNamespace(
            Anthropic=FakeAnthropic,
            RateLimitError=FakeRateLimit,
            APIStatusError=FakeStatusError,
            APIError=FakeAPIError,
        ),
    )
    monkeypatch.setattr(claude.time, "sleep", lambda _: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    for stale in ("ANTHROPIC_EFFORT", "MAL_IMAGE_MAX_EDGE", "SAM3_AGENT_IMAGE_MAX_EDGE",
                  "MAL_MAX_IMAGES_PER_REQUEST", "SAM3_MAX_IMAGES_PER_REQUEST"):
        monkeypatch.delenv(stale, raising=False)
    return types.SimpleNamespace(calls=calls, script=script)


MSGS = [{"role": "user", "content": "hello"}]


class TestHelpers:
    @pytest.mark.parametrize(
        "current,floor,expected",
        [(1024, 384, 819), (400, 384, 384), (384, 384, None), (None, 384, None), (100, 384, None)],
    )
    def test_downscale_ladder(self, current, floor, expected):
        assert _next_smaller_image_edge(current, floor) == expected

    def test_extract_text_joins_blocks_and_ignores_others(self):
        response = types.SimpleNamespace(
            content=[
                types.SimpleNamespace(type="text", text="one"),
                types.SimpleNamespace(type="thinking", text="hidden"),
                {"type": "text", "text": "two"},
            ]
        )
        assert extract_text(response) == "one\ntwo"

    def test_extract_text_of_nothing_is_none(self):
        assert extract_text(types.SimpleNamespace(content=[])) is None
        assert extract_text(types.SimpleNamespace(content=[
            types.SimpleNamespace(type="text", text="   ")])) is None

    def test_env_int_prefers_the_new_name_then_the_legacy_alias(self, monkeypatch):
        monkeypatch.delenv("NEW_NAME", raising=False)
        monkeypatch.setenv("OLD_NAME", "7")
        assert _env_int("NEW_NAME", "OLD_NAME", default=1) == 7
        monkeypatch.setenv("NEW_NAME", "9")
        assert _env_int("NEW_NAME", "OLD_NAME", default=1) == 9
        monkeypatch.setenv("NEW_NAME", "garbage")
        assert _env_int("NEW_NAME", "OLD_NAME", default=1) == 7


class TestGuards:
    def test_missing_api_key_returns_none_without_calling(self, transport, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert claude.send_claude_request(MSGS, api_key=None) is None
        assert transport.calls == []

    def test_invalid_effort_is_refused(self, transport):
        assert claude.send_claude_request(MSGS, effort="turbo") is None
        assert transport.calls == []

    def test_no_convertible_messages_returns_none(self, transport):
        assert claude.send_claude_request([{"role": "system", "content": "only system"}]) is None
        assert transport.calls == []


class TestRequestShape:
    def test_system_prompt_gets_an_ephemeral_cache_breakpoint(self, transport):
        claude.send_claude_request(
            [{"role": "system", "content": "preamble"}, {"role": "user", "content": "hi"}]
        )
        system = transport.calls[0]["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert system[0]["text"] == "preamble"

    def test_cache_can_be_disabled(self, transport):
        claude.send_claude_request(
            [{"role": "system", "content": "preamble"}, {"role": "user", "content": "hi"}],
            enable_prompt_cache=False,
        )
        assert transport.calls[0]["system"] == "preamble"

    def test_effort_is_forwarded_only_when_set(self, transport):
        claude.send_claude_request(MSGS, effort="medium")
        assert transport.calls[0]["output_config"] == {"effort": "medium"}
        claude.send_claude_request(MSGS)
        assert "output_config" not in transport.calls[1]

    def test_max_tokens_is_floored_at_one(self, transport):
        claude.send_claude_request(MSGS, max_tokens=0)
        assert transport.calls[0]["max_tokens"] == 1


class TestRetryPolicy:
    def test_rate_limit_is_retried_then_succeeds(self, transport):
        transport.script.extend([FakeRateLimit("slow down"), _text_response("recovered")])
        assert claude.send_claude_request(MSGS) == "recovered"
        assert len(transport.calls) == 2

    def test_server_error_is_retried(self, transport):
        transport.script.extend([FakeStatusError("boom", 503), _text_response("ok")])
        assert claude.send_claude_request(MSGS) == "ok"

    def test_client_error_is_not_retried(self, transport):
        transport.script.append(FakeStatusError("bad request", 400))
        assert claude.send_claude_request(MSGS) is None
        assert len(transport.calls) == 1

    def test_too_large_falls_back_to_a_single_image_after_the_edge_floor(self, transport):
        transport.script.extend([FakeStatusError("prompt is too long", 400)] * 4)
        transport.script.append(_text_response("fitted"))
        assert claude.send_claude_request(MSGS, max_retries=8) == "fitted"
        assert len(transport.calls) == 5

    def test_empty_response_is_retried_then_gives_up_with_none(self, transport):
        transport.script.extend([_empty_response()] * 3)
        assert claude.send_claude_request(MSGS, max_retries=3) is None
        assert len(transport.calls) == 3

    def test_unexpected_error_returns_none_immediately(self, transport):
        transport.script.append(ValueError("something else"))
        assert claude.send_claude_request(MSGS) is None
        assert len(transport.calls) == 1

    def test_exhausting_retries_returns_none(self, transport):
        transport.script.extend([FakeRateLimit("nope")] * 3)
        assert claude.send_claude_request(MSGS, max_retries=3) is None
        assert len(transport.calls) == 3


class TestTruncation:
    """A response cut off at max_tokens parses as "nothing found".

    The parsers are lenient by design, so an incomplete answer tag yields an
    empty result rather than an error. Believing it would present as poor
    recall, which is why the transport treats it as a failed call.
    """

    @staticmethod
    def _truncated(text="partial <answer>{\"missed_creat"):
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=text)],
            stop_reason="max_tokens",
            usage=types.SimpleNamespace(output_tokens=1024),
        )

    def test_a_truncated_response_is_retried_with_a_doubled_budget(self, transport):
        transport.script.extend([self._truncated(), _text_response("complete")])
        assert claude.send_claude_request(MSGS, max_tokens=1024) == "complete"
        assert [c["max_tokens"] for c in transport.calls] == [1024, 2048]

    def test_the_budget_keeps_doubling_across_retries(self, transport):
        transport.script.extend([self._truncated()] * 3 + [_text_response("ok")])
        claude.send_claude_request(MSGS, max_tokens=1024, max_retries=6)
        assert [c["max_tokens"] for c in transport.calls] == [1024, 2048, 4096, 8192]

    def test_it_stops_at_the_ceiling_and_returns_what_it_has(self, transport):
        transport.script.extend([self._truncated("salvage")] * 2)
        out = claude.send_claude_request(
            MSGS, max_tokens=claude.MAX_COMPLETION_TOKENS, max_retries=2
        )
        assert out == "salvage"
        assert len(transport.calls) == 1

    def test_on_the_final_attempt_the_text_is_returned_rather_than_lost(self, transport):
        transport.script.append(self._truncated("last chance"))
        assert claude.send_claude_request(MSGS, max_tokens=1024, max_retries=1) == "last chance"

    def test_an_untruncated_response_is_not_retried(self, transport):
        transport.script.append(_text_response("fine"))
        assert claude.send_claude_request(MSGS, max_tokens=1024) == "fine"
        assert len(transport.calls) == 1
