"""The per-frame stage graph: ordering, accumulation and stopping."""
from __future__ import annotations

import numpy as np

from marine_autolabel.pipeline.frame import FrameStages, process_frame

FRAME = np.zeros((100, 200, 3), dtype=np.uint8)


def group(gid=1, x=0.5, y=0.5, description="a crab"):
    return {"id": gid, "description": description, "clicks": [{"x": x, "y": y, "label": 1}]}


def mask(seed=1):
    m = np.zeros((100, 200), dtype=bool)
    m[seed : seed + 5, seed : seed + 5] = True
    return {"mask": m}


def stages(**kwargs):
    return FrameStages(**kwargs)


class TestQualityScreen:
    def test_a_dropped_frame_costs_nothing_further(self):
        """rank01's chosen frames are all black and are dropped here."""
        called = []
        outcome = process_frame(
            "rank01", FRAME, pass_count=2,
            stages=stages(
                screen_quality=lambda f: "all_black",
                discover=lambda **kw: called.append(kw) or [group()],
            ),
        )
        assert outcome.skipped_reason == "all_black"
        assert outcome.n_accepted == 0
        assert called == [], "discovery must not run on a dropped frame"

    def test_a_kept_frame_proceeds(self):
        outcome = process_frame(
            "ok", FRAME, pass_count=1,
            stages=stages(screen_quality=lambda f: None, discover=lambda **kw: []),
        )
        assert outcome.skipped_reason is None

    def test_no_screen_means_every_frame_proceeds(self):
        outcome = process_frame("ok", FRAME, pass_count=1, stages=stages())
        assert outcome.skipped_reason is None


class TestPassSequencing:
    def test_a_fixed_sweep_runs_exactly_that_many_passes(self):
        seen = []
        process_frame(
            "f", FRAME, pass_count=3,
            stages=stages(discover=lambda **kw: seen.append(kw["pass_index"]) or []),
        )
        assert seen == [1, 2, 3]

    def test_each_pass_gets_its_own_region(self):
        regions = []
        process_frame(
            "f", FRAME, pass_count=2,
            stages=stages(discover=lambda **kw: regions.append(kw["region"][4]) or []),
        )
        assert regions == ["left half", "right half"]

    def test_border_scan_is_scheduled_on_the_last_pass(self):
        flags = []
        process_frame(
            "f", FRAME, pass_count=3, border_scan="last",
            stages=stages(discover=lambda **kw: flags.append(kw["border_scan"]) or []),
        )
        assert flags == [False, False, True]

    def test_an_empty_pass_does_not_stop_a_fixed_sweep(self):
        seen = []
        process_frame(
            "f", FRAME, pass_count=3,
            stages=stages(discover=lambda **kw: seen.append(kw["pass_index"]) or []),
        )
        assert seen == [1, 2, 3], "a fixed sweep completes its tiling"


class TestConvergence:
    def test_an_empty_proposal_stops_convergence(self):
        calls = {"n": 0}

        def discover(**kw):
            calls["n"] += 1
            return [group()] if calls["n"] < 3 else []

        outcome = process_frame(
            "f", FRAME, pass_count=0,
            stages=stages(discover=discover, recover=lambda **kw: {"recovered": [mask()]}),
        )
        assert calls["n"] == 3
        assert outcome.passes[-1]["stopped"] is True
        assert not outcome.hit_pass_ceiling

    def test_a_model_that_never_stops_hits_the_ceiling(self):
        """Without a ceiling this bills indefinitely, like the repair loop did."""
        calls = {"n": 0}

        def discover(**kw):
            calls["n"] += 1
            return [group(gid=calls["n"], x=0.1 * (calls["n"] % 9) + 0.05)]

        outcome = process_frame(
            "f", FRAME, pass_count=0, max_convergence_passes=5,
            stages=stages(discover=discover, recover=lambda **kw: {"recovered": []}),
        )
        assert calls["n"] == 5
        assert outcome.hit_pass_ceiling is True

    def test_convergence_tiles_before_reviewing_the_whole_frame(self):
        regions = []
        process_frame(
            "f", FRAME, pass_count=0, max_convergence_passes=6,
            stages=stages(
                discover=lambda **kw: regions.append(kw["region"][4]) or [group()],
                recover=lambda **kw: {"recovered": []},
            ),
        )
        assert all("grid tile" in r for r in regions[:4])
        assert regions[4] == "the entire residual frame"


class TestMaskAccumulation:
    def test_accepted_masks_carry_into_later_passes(self):
        """Later passes must see current coverage or they re-find the same animal."""
        seen_counts = []

        def discover(**kw):
            seen_counts.append(len(kw["known_masks"]))
            # A distinct proposal each pass, or repeat-suppression would (correctly)
            # skip it and nothing new would be recovered.
            return [group(gid=kw["pass_index"], x=0.2 * kw["pass_index"])]

        process_frame(
            "f", FRAME, pass_count=3,
            stages=stages(discover=discover, recover=lambda **kw: {"recovered": [mask()]}),
        )
        assert seen_counts == [0, 1, 2]

    def test_prior_known_masks_seed_the_first_pass(self):
        seen = []
        process_frame(
            "f", FRAME, pass_count=1, known_masks=[mask(1), mask(20)],
            stages=stages(discover=lambda **kw: seen.append(len(kw["known_masks"])) or []),
        )
        assert seen == [1 * 2]

    def test_the_outcome_includes_both_prior_and_recovered_masks(self):
        outcome = process_frame(
            "f", FRAME, pass_count=1, known_masks=[mask(1)],
            stages=stages(
                discover=lambda **kw: [group()],
                recover=lambda **kw: {"recovered": [mask(20), mask(40)]},
            ),
        )
        assert outcome.n_accepted == 3

    def test_the_caller_known_masks_list_is_not_mutated(self):
        known = [mask(1)]
        process_frame(
            "f", FRAME, pass_count=1, known_masks=known,
            stages=stages(
                discover=lambda **kw: [group()],
                recover=lambda **kw: {"recovered": [mask(20)]},
            ),
        )
        assert len(known) == 1


class TestRepeatSuppression:
    def test_the_same_proposal_twice_is_not_run_again(self):
        runs = []

        def recover(**kw):
            runs.append(len(kw["groups"]))
            return {"recovered": []}

        outcome = process_frame(
            "f", FRAME, pass_count=2,
            stages=stages(discover=lambda **kw: [group()], recover=recover),
        )
        assert runs == [1], "the second pass proposes the same thing and is skipped"
        assert outcome.passes[1]["n_repeat_skipped"] == 1

    def test_nearby_duplicate_clicks_are_deduped_within_a_pass(self):
        proposals = [group(1, 0.50, 0.50), group(2, 0.505, 0.50), group(3, 0.90, 0.90)]
        outcome = process_frame(
            "f", FRAME, pass_count=1, click_dedup_px=40,
            stages=stages(
                discover=lambda **kw: proposals, recover=lambda **kw: {"recovered": []}
            ),
        )
        assert outcome.passes[0]["n_dedup_removed"] == 1
        assert outcome.passes[0]["n_run"] == 2

    def test_a_different_creature_at_the_same_place_still_runs(self):
        calls = {"n": 0}

        def discover(**kw):
            calls["n"] += 1
            return [group(description="tan crab" if calls["n"] == 1 else "orange star")]

        outcome = process_frame(
            "f", FRAME, pass_count=2,
            stages=stages(discover=discover, recover=lambda **kw: {"recovered": []}),
        )
        assert outcome.passes[1]["n_run"] == 1


class TestReporting:
    def test_the_summary_explains_each_pass(self):
        outcome = process_frame(
            "f", FRAME, pass_count=2,
            stages=stages(
                discover=lambda **kw: [group(gid=kw["pass_index"], x=0.1 * kw["pass_index"])],
                recover=lambda **kw: {"recovered": [mask()], "hit_round_cap": False},
            ),
        )
        summary = outcome.summary()
        assert summary["n_passes"] == 2
        assert summary["n_accepted"] == 2
        assert all("region" in p for p in summary["passes"])

    def test_a_capped_repair_round_is_surfaced(self):
        outcome = process_frame(
            "f", FRAME, pass_count=1,
            stages=stages(
                discover=lambda **kw: [group()],
                recover=lambda **kw: {"recovered": [], "hit_round_cap": True},
            ),
        )
        assert outcome.passes[0]["hit_round_cap"] is True


class TestConsolidation:
    def test_consolidation_runs_once_at_the_end(self):
        calls = []

        def consolidate(masks):
            calls.append(len(masks))
            return masks[:1]

        outcome = process_frame(
            "f", FRAME, pass_count=2,
            stages=stages(
                discover=lambda **kw: [group(gid=kw["pass_index"], x=0.1 * kw["pass_index"])],
                recover=lambda **kw: {"recovered": [mask()]},
                consolidate=consolidate,
            ),
        )
        assert calls == [2], "once, over the full accepted set"
        assert outcome.n_accepted == 1

    def test_consolidation_is_skipped_for_a_dropped_frame(self):
        calls = []
        process_frame(
            "f", FRAME, pass_count=1,
            stages=stages(screen_quality=lambda f: "black", consolidate=calls.append),
        )
        assert calls == []


class TestAttritionReporting:
    """A pass that proposes many and keeps few must say where they went."""

    def test_the_breakdown_is_carried_into_the_pass_record(self):
        outcome = process_frame(
            "f", FRAME, pass_count=1,
            stages=stages(
                discover=lambda **kw: [group(gid=i, x=0.1 * i) for i in range(1, 5)],
                recover=lambda **kw: {
                    "recovered": [mask()],
                    "n_generated": 4,
                    "n_verify_kept": 2,
                    "n_verify_dropped": 2,
                    "n_repair_recovered": 1,
                    "mask_nms_removed": 1,
                    "low_confidence": [mask(2)],
                    "repair_rounds": 1,
                    "hit_round_cap": False,
                },
            ),
        )
        record = outcome.passes[0]
        assert record["n_proposed"] == 4
        assert record["n_generated"] == 4
        assert record["n_verify_kept"] == 2
        assert record["n_verify_dropped"] == 2
        assert record["n_repair_recovered"] == 1
        assert record["n_nms_removed"] == 1
        assert record["n_low_confidence"] == 1
        assert record["repair_rounds"] == 1

    def test_a_recover_stage_that_reports_nothing_still_works(self):
        outcome = process_frame(
            "f", FRAME, pass_count=1,
            stages=stages(
                discover=lambda **kw: [group()], recover=lambda **kw: {"recovered": []}
            ),
        )
        assert outcome.passes[0]["n_recovered"] == 0
        assert outcome.passes[0]["n_low_confidence"] == 0
