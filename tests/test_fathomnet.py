"""FathomNet baseline detection handling."""
from __future__ import annotations

import types

import pytest

from marine_autolabel.benchmarks.fathomnet import (
    MODELS,
    Detection,
    extract_detections,
    filter_by_confidence,
    summarize,
    threshold_slug,
)


def result(boxes=None, names=None):
    if boxes is None:
        return types.SimpleNamespace(boxes=None, names={})
    return types.SimpleNamespace(
        boxes=types.SimpleNamespace(
            xyxy=[b[0] for b in boxes],
            conf=[b[1] for b in boxes],
            cls=[b[2] for b in boxes],
        ),
        names=names or {0: "fish", 1: "coral"},
    )


class TestRegistry:
    def test_the_published_baselines_are_listed(self):
        assert "fathomnet_mbari_315k_yolov8" in MODELS
        assert "fathomnet_benthic_2025" in MODELS

    def test_every_entry_can_be_fetched(self):
        for key, spec in MODELS.items():
            assert spec["repo_id"].startswith("FathomNet/"), key
            assert spec["filename"].endswith(".pt"), key
            assert spec["display_name"], key


class TestExtraction:
    def test_pulls_boxes_scores_and_class_names(self):
        detections = extract_detections(result([([1, 2, 3, 4], 0.9, 0)]))
        assert len(detections) == 1
        assert detections[0].class_name == "fish"
        assert detections[0].confidence == 0.9
        assert detections[0].xyxy == (1.0, 2.0, 3.0, 4.0)

    def test_no_boxes_attribute_is_empty_not_an_error(self):
        assert extract_detections(result()) == []

    def test_an_empty_detection_set_is_legitimate(self):
        """The frame simply contains nothing from this detector's classes."""
        assert extract_detections(result([])) == []

    def test_a_malformed_result_is_refused_rather_than_truncated(self):
        bad = types.SimpleNamespace(
            boxes=types.SimpleNamespace(xyxy=[[1, 2, 3, 4], [5, 6, 7, 8]],
                                        conf=[0.9], cls=[0, 1]),
            names={0: "fish", 1: "coral"},
        )
        with pytest.raises(ValueError):
            extract_detections(bad)

    def test_tensor_like_values_are_converted(self):
        class Fake:
            def __init__(self, data):
                self._data = data

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self._data

        res = types.SimpleNamespace(
            boxes=types.SimpleNamespace(
                xyxy=Fake([[1, 2, 3, 4]]), conf=Fake([0.5]), cls=Fake([1])
            ),
            names={1: "coral"},
        )
        assert extract_detections(res)[0].class_name == "coral"

    def test_serialisation_rounds_sensibly(self):
        d = Detection((1.23456, 2.0, 3.0, 4.0), 0.123456789, 0, "fish")
        assert d.as_dict()["xyxy"][0] == 1.235
        assert d.as_dict()["confidence"] == 0.123457


class TestThresholding:
    DETECTIONS = [
        Detection((0, 0, 1, 1), 0.9, 0, "fish"),
        Detection((0, 0, 1, 1), 0.5, 0, "fish"),
        Detection((0, 0, 1, 1), 0.2, 1, "coral"),
    ]

    def test_filters_below_the_threshold(self):
        assert len(filter_by_confidence(self.DETECTIONS, 0.5)) == 2

    def test_the_threshold_is_inclusive(self):
        assert len(filter_by_confidence(self.DETECTIONS, 0.9)) == 1

    def test_results_are_ordered_by_confidence(self):
        kept = filter_by_confidence(self.DETECTIONS, 0.0)
        assert [d.confidence for d in kept] == [0.9, 0.5, 0.2]

    @pytest.mark.parametrize(
        "value,expected", [(0.25, "conf25"), (0.5, "conf50"), (0.05, "conf05"), (1.0, "conf100")]
    )
    def test_threshold_slug_is_filesystem_safe(self, value, expected):
        slug = threshold_slug(value)
        assert slug == expected
        assert "." not in slug


class TestSummary:
    def test_counts_per_class(self):
        out = summarize(TestThresholding.DETECTIONS, 0.0)
        assert out["n_detections"] == 3
        assert out["per_class"] == {"coral": 1, "fish": 2}
        assert out["n_classes"] == 2

    def test_the_threshold_is_applied_before_counting(self):
        out = summarize(TestThresholding.DETECTIONS, 0.6)
        assert out["n_detections"] == 1
        assert out["per_class"] == {"fish": 1}

    def test_an_empty_result_summarises_to_zero_not_an_error(self):
        out = summarize([], 0.5)
        assert out["n_detections"] == 0
        assert out["max_confidence"] == 0.0
        assert out["per_class"] == {}
