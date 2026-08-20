"""Repeat aggregation and the guards that keep the numbers comparable."""
from __future__ import annotations

import json

import pytest

from marine_autolabel.eval.repeats import (
    check_comparable,
    describe,
    format_mean_std,
    load_repeats,
    summarize,
)


def doc(n_final, frames=("a", "b"), model="claude-opus-5", runtime=100.0):
    return {
        "config": {"model": model, "mask_generator": "hybrid"},
        "n_final": n_final,
        "runtime_sec": runtime,
        "frames": [{"frame_id": f, "n_final": n_final, "n_recovered": n_final - 1}
                   for f in frames],
    }


def write_repeats(root, docs, prefix="repeat"):
    for i, d in enumerate(docs, 1):
        p = root / f"{prefix}_{i}"
        p.mkdir(parents=True, exist_ok=True)
        (p / "summary.json").write_text(json.dumps(d))
    return root


class TestDescribe:
    def test_reports_mean_std_min_max(self):
        stats = describe([2.0, 4.0, 6.0])
        assert stats["mean"] == 4.0
        assert stats["min"] == 2.0 and stats["max"] == 6.0
        assert stats["std"] > 0

    def test_a_single_value_has_zero_spread(self):
        assert describe([5.0])["std"] == 0.0

    def test_formats_as_mean_plus_minus_std(self):
        assert format_mean_std({"mean": 0.5, "std": 0.125}, places=3) == "0.500 +/- 0.125"


class TestRepeatCount:
    def test_three_repeats_are_accepted(self, tmp_path):
        write_repeats(tmp_path, [doc(5), doc(6), doc(7)])
        assert len(load_repeats(tmp_path)) == 3

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_fewer_than_three_is_refused(self, tmp_path, n):
        """A single run is a plumbing check, not a result."""
        write_repeats(tmp_path, [doc(5)] * n)
        with pytest.raises(ValueError, match="at least 3 repeats"):
            load_repeats(tmp_path)

    def test_the_error_explains_why(self, tmp_path):
        with pytest.raises(ValueError, match="plumbing check, not a result"):
            load_repeats(tmp_path)


class TestComparability:
    def test_matching_runs_pass(self):
        check_comparable([doc(5), doc(6), doc(7)])

    def test_a_changed_model_is_refused(self):
        """Averaging across a model change describes neither configuration."""
        with pytest.raises(ValueError, match="different\\s+experiments"):
            check_comparable([doc(5), doc(6, model="claude-sonnet-5"), doc(7)])

    def test_a_changed_frame_set_is_refused(self):
        with pytest.raises(ValueError, match="same ordered frame IDs"):
            check_comparable([doc(5), doc(6, frames=("a", "c")), doc(7)])

    def test_frame_order_matters(self):
        with pytest.raises(ValueError, match="same ordered frame IDs"):
            check_comparable([doc(5), doc(6, frames=("b", "a")), doc(7)])


class TestSummarize:
    def test_aggregates_run_level_metrics(self, tmp_path):
        write_repeats(tmp_path, [doc(4), doc(6), doc(8)])
        out = summarize(tmp_path)
        assert out["n_repeats"] == 3
        assert out["metrics"]["n_final"]["mean"] == 6.0
        assert out["metrics"]["n_final"]["min"] == 4.0

    def test_aggregates_per_frame(self, tmp_path):
        write_repeats(tmp_path, [doc(4), doc(6), doc(8)])
        out = summarize(tmp_path)
        assert set(out["per_frame"]) == {"a", "b"}
        assert out["per_frame"]["a"]["n_recovered"]["mean"] == 5.0

    def test_a_missing_metric_is_skipped_not_faked(self, tmp_path):
        write_repeats(tmp_path, [doc(4), doc(6), doc(8)])
        out = summarize(tmp_path)
        assert "n_text_verified" not in out["metrics"]

    def test_spread_is_reported_not_hidden(self, tmp_path):
        write_repeats(tmp_path, [doc(1), doc(6), doc(20)])
        assert summarize(tmp_path)["metrics"]["n_final"]["std"] > 5
