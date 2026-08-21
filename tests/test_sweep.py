"""Discovery sweep scheduling."""
from __future__ import annotations

import pytest

from marine_autolabel.clickengine.sweep import (
    choose_mask_guided_pass_count,
    click_counts,
    discovery_focus_region,
    filter_prior_attempt_groups,
    first_positive_click,
    should_run_border_scan,
)


def group(description="tan crab", clicks=None):
    return {
        "description": description,
        "clicks": clicks if clicks is not None else [{"x": 0.5, "y": 0.5, "label": 1}],
    }


def attempt(x=0.5, y=0.5, description="tan crab"):
    return {"click": {"x": x, "y": y}, "description": description}


class TestPassCount:
    def test_a_sparse_scene_earns_an_extra_pass(self):
        """empty: the first pass found zero objects."""
        assert choose_mask_guided_pass_count(
            3, initial_known_count=0, sparse_extra_pass=True, finder_mode="mask-guided"
        ) == 4

    def test_no_extra_pass_when_the_first_pass_found_something(self):
        assert choose_mask_guided_pass_count(
            3, initial_known_count=2, sparse_extra_pass=True, finder_mode="mask-guided"
        ) == 3

    def test_no_extra_pass_when_disabled(self):
        assert choose_mask_guided_pass_count(
            3, initial_known_count=0, sparse_extra_pass=False, finder_mode="mask-guided"
        ) == 3

    def test_the_cold_finder_gets_no_extra_pass(self):
        assert choose_mask_guided_pass_count(
            3, initial_known_count=0, sparse_extra_pass=True, finder_mode="s3"
        ) == 3

    @pytest.mark.parametrize("requested", [0, -1])
    def test_convergence_mode_is_preserved(self, requested):
        assert choose_mask_guided_pass_count(
            requested, initial_known_count=0, sparse_extra_pass=True, finder_mode="hybrid"
        ) == 0


class TestBorderScan:
    def test_off_never_runs(self):
        assert not should_run_border_scan(
            "off", convergence_mode=False, pass_index=1, requested_passes=3
        )

    def test_every_runs_on_each_pass(self):
        assert all(
            should_run_border_scan("every", convergence_mode=False, pass_index=i,
                                   requested_passes=3)
            for i in (1, 2, 3)
        )

    def test_last_runs_only_on_the_final_pass(self):
        runs = [
            should_run_border_scan("last", convergence_mode=False, pass_index=i,
                                   requested_passes=3)
            for i in (1, 2, 3)
        ]
        assert runs == [False, False, True]

    @pytest.mark.parametrize("mode", ["every", "last"])
    def test_convergence_audits_once_on_pass_one(self, mode):
        """There is no knowable last pass, so 'last' cannot mean anything else."""
        runs = [
            should_run_border_scan(mode, convergence_mode=True, pass_index=i,
                                   requested_passes=0)
            for i in (1, 2, 3, 9)
        ]
        assert runs == [True, False, False, False]

    def test_off_stays_off_in_convergence_mode(self):
        assert not should_run_border_scan(
            "off", convergence_mode=True, pass_index=1, requested_passes=0
        )


class TestFocusRegion:
    def test_a_single_pass_sees_everything(self):
        left, right, top, bottom, label = discovery_focus_region(1, 1)
        assert (left, right, top, bottom) == (0.0, 1.0, 0.0, 1.0)
        assert label == "the entire frame"

    def test_two_passes_split_left_and_right(self):
        assert discovery_focus_region(1, 2)[:2] == (0.0, 0.5)
        assert discovery_focus_region(2, 2)[:2] == (0.5, 1.0)
        assert discovery_focus_region(2, 2)[4] == "right half"

    def test_three_passes_split_into_thirds(self):
        labels = [discovery_focus_region(i, 3)[4] for i in (1, 2, 3)]
        assert labels == ["left third", "middle third", "right third"]

    def test_four_passes_tile_two_by_two(self):
        tiles = [discovery_focus_region(i, 4)[:4] for i in (1, 2, 3, 4)]
        assert tiles == [
            (0.0, 0.5, 0.0, 0.5), (0.5, 1.0, 0.0, 0.5),
            (0.0, 0.5, 0.5, 1.0), (0.5, 1.0, 0.5, 1.0),
        ]

    def test_tiles_cover_the_frame_without_gaps(self):
        for count in (4, 5, 6, 9):
            tiles = [discovery_focus_region(i, count)[:4] for i in range(1, count + 1)]
            assert min(t[0] for t in tiles) == 0.0
            assert max(t[1] for t in tiles) == 1.0
            assert min(t[2] for t in tiles) == 0.0
            assert max(t[3] for t in tiles) == 1.0

    def test_bounds_never_leave_the_frame(self):
        for count in (1, 2, 3, 4, 5, 7, 9, 16):
            for index in range(1, count + 1):
                left, right, top, bottom, _ = discovery_focus_region(index, count)
                assert 0.0 <= left < right <= 1.0
                assert 0.0 <= top < bottom <= 1.0

    def test_convergence_tiles_first_then_reviews_the_whole_frame(self):
        early = [discovery_focus_region(i, 0)[:4] for i in (1, 2, 3, 4)]
        assert early == [discovery_focus_region(i, 4)[:4] for i in (1, 2, 3, 4)]
        assert discovery_focus_region(5, 0)[:4] == (0.0, 1.0, 0.0, 1.0)
        assert discovery_focus_region(99, 0)[4] == "the entire residual frame"

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_pass_index_is_refused(self, bad):
        """The original produced negative bounds instead of complaining."""
        with pytest.raises(ValueError, match="1-based"):
            discovery_focus_region(bad, 4)

    def test_an_out_of_range_index_is_refused(self):
        with pytest.raises(ValueError, match="exceeds"):
            discovery_focus_region(4, 3)


class TestPriorAttemptFilter:
    def test_a_fresh_proposal_runs(self):
        kept, skipped = filter_prior_attempt_groups([group()], [])
        assert len(kept) == 1 and skipped == []

    def test_the_same_creature_at_the_same_place_is_skipped(self):
        kept, skipped = filter_prior_attempt_groups([group()], [attempt()])
        assert kept == [] and len(skipped) == 1

    def test_a_different_creature_at_the_same_place_still_runs(self):
        kept, _ = filter_prior_attempt_groups(
            [group(description="orange sea star")], [attempt(description="tan crab")]
        )
        assert len(kept) == 1

    def test_the_same_creature_elsewhere_still_runs(self):
        kept, _ = filter_prior_attempt_groups([group()], [attempt(x=0.9, y=0.9)])
        assert len(kept) == 1, "a materially different click is not a repeat"

    def test_spatial_tolerance_is_configurable(self):
        groups, attempts = [group()], [attempt(x=0.53)]
        assert filter_prior_attempt_groups(groups, attempts, spatial_tolerance=0.01)[0]
        assert filter_prior_attempt_groups(groups, attempts, spatial_tolerance=0.10)[0] == []

    def test_short_tokens_are_ignored_when_comparing_descriptions(self):
        """'a', 'of' and the like would otherwise inflate overlap."""
        kept, _ = filter_prior_attempt_groups(
            [group(description="a big red anemone")], [attempt(description="a big blue crab")]
        )
        assert len(kept) == 1

    def test_an_empty_description_never_matches(self):
        kept, _ = filter_prior_attempt_groups([group(description="")], [attempt()])
        assert len(kept) == 1

    def test_a_seedless_group_is_skipped_not_raised(self):
        """The parser drops these, but a mid-run raise would abort a fan-out."""
        seedless = group(clicks=[{"x": 0.5, "y": 0.5, "label": 0}])
        kept, skipped = filter_prior_attempt_groups([seedless], [])
        assert kept == [] and skipped == [seedless]

    def test_every_group_is_accounted_for(self):
        groups = [group(), group(description="x"), group(clicks=[])]
        kept, skipped = filter_prior_attempt_groups(groups, [attempt()])
        assert len(kept) + len(skipped) == len(groups)


class TestClickCounts:
    def test_counts_by_label(self):
        clicks = [
            {"x": 0, "y": 0, "label": 1}, {"x": 0, "y": 0, "label": 1},
            {"x": 0, "y": 0, "label": 0},
        ]
        assert click_counts(clicks) == {"positive": 2, "negative": 1}

    def test_empty(self):
        assert click_counts([]) == {"positive": 0, "negative": 0}


class TestFirstPositiveClick:
    def test_returns_the_first_foreground_click(self):
        clicks = [{"x": 0.1, "y": 0.1, "label": 0}, {"x": 0.2, "y": 0.2, "label": 1}]
        assert first_positive_click({"clicks": clicks})["x"] == 0.2

    def test_returns_none_when_there_is_no_seed(self):
        assert first_positive_click({"clicks": [{"x": 0.1, "y": 0.1, "label": 0}]}) is None
        assert first_positive_click({}) is None

    def test_returns_a_copy(self):
        clicks = [{"x": 0.2, "y": 0.2, "label": 1}]
        first_positive_click({"clicks": clicks})["x"] = 9.0
        assert clicks[0]["x"] == 0.2


class TestDescriptionAwareDedup:
    """Proximity alone must not collapse two distinct organisms.

    Live failures: a cap-and-stem organism 39.7 px from a whip's seed, and a
    translucent whip near a branching coral, both removed by the pure 40 px
    rule in dense scenes.
    """

    @staticmethod
    def _group(x, y, description):
        return {"id": 0, "description": description,
                "clicks": [{"x": x, "y": y, "label": 1}]}

    def test_nearby_distinct_organisms_both_survive(self):
        from marine_autolabel.clickengine.sweep import dedup_proposals

        groups = [
            self._group(0.573, 0.417, "thin bright vertical sea whip"),
            self._group(0.604, 0.451, "bright cap-and-stem mushroom sponge"),
        ]
        kept, removed = dedup_proposals(groups, 1280, 720, px=40)
        assert removed == 0 and len(kept) == 2

    def test_two_markers_on_the_same_animal_still_collapse(self):
        from marine_autolabel.clickengine.sweep import dedup_proposals

        groups = [
            self._group(0.500, 0.500, "large tan bushy coral colony"),
            self._group(0.510, 0.505, "tan bushy coral colony"),
        ]
        kept, removed = dedup_proposals(groups, 1280, 720, px=40)
        assert removed == 1 and len(kept) == 1

    def test_distant_same_description_groups_are_kept(self):
        from marine_autolabel.clickengine.sweep import dedup_proposals

        groups = [
            self._group(0.1, 0.1, "sea whip"),
            self._group(0.9, 0.9, "sea whip"),
        ]
        kept, removed = dedup_proposals(groups, 1280, 720, px=40)
        assert removed == 0 and len(kept) == 2

    def test_empty_descriptions_defer_to_proximity(self):
        from marine_autolabel.clickengine.sweep import dedup_proposals

        groups = [self._group(0.5, 0.5, ""), self._group(0.505, 0.5, "")]
        kept, removed = dedup_proposals(groups, 1280, 720, px=40)
        assert removed == 1, "no description evidence -> the old behaviour"

    def test_survivors_are_renumbered(self):
        from marine_autolabel.clickengine.sweep import dedup_proposals

        groups = [
            self._group(0.1, 0.1, "a"), self._group(0.5, 0.5, "b"),
        ]
        kept, _ = dedup_proposals(groups, 1280, 720)
        assert [g["id"] for g in kept] == [1, 2]
