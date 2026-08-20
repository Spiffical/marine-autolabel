"""SeaTube annotation matching: closed vocabulary and majority consensus."""
from __future__ import annotations

import pytest

from marine_autolabel.eval.seatube import (
    build_consensus,
    compress_id_ranges,
    int_list,
    normalize_response,
    response_object_choices,
)

ANNOTATIONS = [
    {"annotation_id": 11, "taxon": "Chrysogorgia"},
    {"annotation_id": 12, "taxon": "Walteria"},
]


def match(object_ids, annotation_ids, confidence=0.9, reason=""):
    return {
        "object_ids": object_ids,
        "annotation_ids": annotation_ids,
        "confidence": confidence,
        "reason": reason,
    }


def normalized(matches, **extra):
    return normalize_response(
        {"matches": matches, **extra}, object_count=5, annotations=ANNOTATIONS
    )


class TestIntList:
    @pytest.mark.parametrize(
        "value,expected",
        [(3, [3]), ([1, 2], [1, 2]), (["4", 5], [4, 5]), ("x", []), (None, []), ([True], [])],
    )
    def test_coercion(self, value, expected):
        assert int_list(value) == expected


class TestNormalize:
    def test_a_valid_match_survives(self):
        out = normalized([match([1, 2], [11])])
        assert out["matches"][0]["object_ids"] == [1, 2]
        assert out["matches"][0]["taxa"] == ["Chrysogorgia"]

    def test_an_unoffered_annotation_id_is_discarded(self):
        """The matcher never invents taxonomy: ids it was not given are dropped."""
        assert normalized([match([1], [99])])["matches"] == []

    def test_an_out_of_range_object_id_is_discarded(self):
        assert normalized([match([99], [11])])["matches"] == []

    def test_a_partially_valid_match_keeps_only_valid_ids(self):
        out = normalized([match([1, 99], [11, 99])])
        assert out["matches"][0]["object_ids"] == [1]
        assert out["matches"][0]["annotation_ids"] == [11]

    def test_singular_key_names_are_accepted(self):
        out = normalize_response(
            {"matches": [{"object_id": 2, "annotation_id": 12, "confidence": 0.8}]},
            object_count=5, annotations=ANNOTATIONS,
        )
        assert out["matches"][0]["object_ids"] == [2]

    def test_confidence_is_clamped(self):
        assert normalized([match([1], [11], confidence=5)])["matches"][0]["confidence"] == 1.0
        assert normalized([match([1], [11], confidence=-1)])["matches"][0]["confidence"] == 0.0

    def test_an_unparseable_confidence_becomes_zero(self):
        assert normalized([match([1], [11], confidence="high")])["matches"][0]["confidence"] == 0.0

    def test_unmatched_lists_are_filtered_too(self):
        out = normalized([], unmatched_object_ids=[1, 99], unmatched_annotation_ids=[12, 99])
        assert out["unmatched_object_ids"] == [1]
        assert out["unmatched_annotation_ids"] == [12]

    def test_a_non_dict_response_is_none(self):
        assert normalize_response(None, object_count=5, annotations=ANNOTATIONS) is None


class TestObjectChoices:
    def test_the_highest_confidence_match_wins_per_object(self):
        response = normalized([match([1], [11], 0.4), match([1], [12], 0.9)])
        choices = response_object_choices(response)
        assert choices[1]["taxa"] == ["Walteria"]


class TestConsensus:
    @staticmethod
    def _responses(*taxa_per_repeat, confidence=0.9):
        out = []
        for annotation_id in taxa_per_repeat:
            out.append(normalized([match([1], [annotation_id], confidence)]))
        return out

    def test_a_unanimous_label_is_assigned(self):
        out = build_consensus(
            self._responses(11, 11, 11), object_count=1,
            annotations=ANNOTATIONS, configured_repeats=3,
        )
        assert out["assignments"][0]["label"] == "Chrysogorgia"
        assert out["assignments"][0]["support"] == 3

    def test_a_majority_carries(self):
        out = build_consensus(
            self._responses(11, 11, 12), object_count=1,
            annotations=ANNOTATIONS, configured_repeats=3,
        )
        assert out["assignments"][0]["label"] == "Chrysogorgia"
        assert out["assignments"][0]["support"] == 2

    def test_no_majority_leaves_the_object_unmatched(self):
        out = build_consensus(
            self._responses(11, 12), object_count=1,
            annotations=ANNOTATIONS, configured_repeats=3,
        )
        assert out["assignments"] == []
        assert out["unmatched_object_ids"] == [1]

    def test_a_lone_surviving_response_cannot_carry_a_majority(self):
        """If two of three calls failed, one opinion must not become an answer."""
        out = build_consensus(
            self._responses(11), object_count=1,
            annotations=ANNOTATIONS, configured_repeats=3,
        )
        assert out["assignments"] == []
        assert out["valid_response_count"] == 1
        assert out["consensus_threshold"] == 2

    def test_low_mean_confidence_blocks_assignment(self):
        out = build_consensus(
            self._responses(11, 11, 11, confidence=0.2), object_count=1,
            annotations=ANNOTATIONS, configured_repeats=3,
        )
        assert out["assignments"] == []

    def test_the_confidence_floor_is_configurable(self):
        out = build_consensus(
            self._responses(11, 11, 11, confidence=0.2), object_count=1,
            annotations=ANNOTATIONS, configured_repeats=3, min_mean_confidence=0.1,
        )
        assert len(out["assignments"]) == 1

    def test_spread_across_repeats_is_reported(self):
        responses = [
            normalized([match([1], [11], c)]) for c in (0.6, 0.9, 1.0)
        ]
        out = build_consensus(
            responses, object_count=1, annotations=ANNOTATIONS, configured_repeats=3
        )
        assert out["assignments"][0]["confidence_std"] > 0

    def test_unmatched_annotations_are_reported(self):
        out = build_consensus(
            self._responses(11, 11, 11), object_count=1,
            annotations=ANNOTATIONS, configured_repeats=3,
        )
        assert out["unmatched_annotation_ids"] == [12]

    def test_objects_never_mentioned_are_unmatched(self):
        out = build_consensus(
            self._responses(11, 11, 11), object_count=3,
            annotations=ANNOTATIONS, configured_repeats=3,
        )
        assert out["unmatched_object_ids"] == [2, 3]

    def test_no_responses_assigns_nothing(self):
        out = build_consensus(
            [], object_count=2, annotations=ANNOTATIONS, configured_repeats=3
        )
        assert out["assignments"] == []
        assert out["unmatched_object_ids"] == [1, 2]


class TestIdRanges:
    @pytest.mark.parametrize(
        "values,expected",
        [
            ([1, 2, 3, 7, 9, 10], "1-3, 7, 9-10"),
            ([5], "5"),
            ([], ""),
            ([3, 1, 2], "1-3"),
            ([1, 1, 2], "1-2"),
        ],
    )
    def test_compression(self, values, expected):
        assert compress_id_ranges(values) == expected
