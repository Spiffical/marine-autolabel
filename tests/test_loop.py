"""Click-loop control rules."""
from __future__ import annotations

import itertools

import pytest

from marine_autolabel.clickengine.loop import (
    bounded_corrected_positive_click,
    click_budget_reached,
    is_reasoning_model,
    iteration_indices,
    response_token_budget,
)


class TestIterationIndices:
    def test_a_positive_ceiling_is_a_finite_range(self):
        assert list(iteration_indices(3)) == [0, 1, 2]

    @pytest.mark.parametrize("ceiling", [0, -1])
    def test_non_positive_means_agent_controlled(self, ceiling):
        first_five = list(itertools.islice(iteration_indices(ceiling), 5))
        assert first_five == [0, 1, 2, 3, 4], "must be an unbounded counter"


class TestClickBudget:
    def test_reached_at_the_ceiling(self):
        assert click_budget_reached([1, 2, 3], 3)

    def test_not_reached_below_it(self):
        assert not click_budget_reached([1, 2], 3)

    @pytest.mark.parametrize("ceiling", [0, -1])
    def test_a_non_positive_ceiling_is_never_reached(self, ceiling):
        assert not click_budget_reached(list(range(100)), ceiling)


class TestTokenBudget:
    @pytest.mark.parametrize(
        "model", ["claude-sonnet-5", "claude-opus-5", "claude-fable-5"]
    )
    def test_every_claude_5_model_is_floored(self, model):
        """A small budget can go entirely to reasoning, truncating the tag.

        fable-5 was omitted from the original list even while it was the click
        model, so its truncations would have looked like poor recall.
        """
        assert response_token_budget(model, 1024) == 4096
        assert is_reasoning_model(model)

    @pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-5"])
    def test_a_larger_request_is_respected(self, model):
        assert response_token_budget(model, 8192) == 8192

    @pytest.mark.parametrize("model", ["claude-sonnet-4-6", "gpt-4o", "qwen3.5"])
    def test_older_models_keep_the_cheaper_cap(self, model):
        assert response_token_budget(model, 1024) == 1024
        assert not is_reasoning_model(model)

    def test_matching_is_case_insensitive(self):
        assert response_token_budget("Claude-Sonnet-5", 512) == 4096

    def test_the_floor_is_configurable(self):
        assert response_token_budget("claude-opus-5", 512, minimum=2048) == 2048


class TestBoundedCorrection:
    BASE = [{"x": 0.50, "y": 0.50, "label": 1}]

    def test_a_small_nudge_is_applied(self):
        clicks, displacement, applied = bounded_corrected_positive_click(
            self.BASE, {"x": 0.52, "y": 0.50, "label": 1}
        )
        assert applied
        assert displacement == pytest.approx(0.02)
        assert clicks[0]["x"] == 0.52

    def test_a_large_move_is_refused_to_avoid_switching_organism(self):
        """In a dense scene a big jump usually lands on a different animal."""
        clicks, displacement, applied = bounded_corrected_positive_click(
            self.BASE, {"x": 0.90, "y": 0.50, "label": 1}
        )
        assert not applied
        assert displacement == pytest.approx(0.40)
        assert clicks == self.BASE, "the original clicks must survive untouched"

    def test_the_boundary_is_inclusive(self):
        _, _, applied = bounded_corrected_positive_click(
            self.BASE, {"x": 0.60, "y": 0.50, "label": 1}, max_displacement=0.10
        )
        assert applied

    def test_displacement_is_measured_from_the_nearest_positive_seed(self):
        clicks = [{"x": 0.10, "y": 0.10, "label": 1}, {"x": 0.80, "y": 0.80, "label": 1}]
        _, displacement, applied = bounded_corrected_positive_click(
            clicks, {"x": 0.82, "y": 0.80, "label": 1}
        )
        assert applied
        assert displacement == pytest.approx(0.02)

    def test_with_no_positive_seed_the_correction_is_prepended(self):
        negatives = [{"x": 0.5, "y": 0.5, "label": 0}]
        clicks, displacement, applied = bounded_corrected_positive_click(
            negatives, {"x": 0.1, "y": 0.1, "label": 1}
        )
        assert applied and displacement is None
        assert clicks[0]["label"] == 1
        assert len(clicks) == 2

    def test_negative_clicks_are_preserved_through_a_correction(self):
        clicks = [{"x": 0.5, "y": 0.5, "label": 1}, {"x": 0.2, "y": 0.2, "label": 0}]
        merged, _, applied = bounded_corrected_positive_click(
            clicks, {"x": 0.52, "y": 0.5, "label": 1}
        )
        assert applied
        assert {"x": 0.2, "y": 0.2, "label": 0} in merged

    def test_the_caller_list_is_never_mutated(self):
        original = [dict(c) for c in self.BASE]
        bounded_corrected_positive_click(original, {"x": 0.99, "y": 0.99, "label": 1})
        assert original == self.BASE
