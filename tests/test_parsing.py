"""MLLM response parsers.

Every parser is lenient: malformed input yields "nothing", never an exception.
Callers pass responses that may be None because the API call failed.
"""
from __future__ import annotations

import json

import pytest

from marine_autolabel.clickengine.parsing import (
    parse_click,
    parse_click_proposals,
    parse_click_refinement_response,
    parse_creature_click_groups,
    parse_frame_validity,
    parse_refinement_response,
    parse_som_response,
)


def answer(obj) -> str:
    return f"some reasoning first\n<answer>{json.dumps(obj)}</answer>"


def clicks(*xy) -> list[dict]:
    return [{"x": x, "y": y, "label": 1} for x, y in xy]


MALFORMED = ["", "no tags here", "<answer>not json</answer>", None, 42, "<answer></answer>"]


class TestParseClick:
    def test_accepts_a_well_formed_click(self):
        assert parse_click({"x": 0.25, "y": 0.75, "label": 1}) == {
            "x": 0.25, "y": 0.75, "label": 1
        }

    def test_label_zero_is_kept(self):
        assert parse_click({"x": 0.5, "y": 0.5, "label": 0})["label"] == 0

    @pytest.mark.parametrize("bad", [-0.01, 1.01, "0.5", None, float("nan")])
    def test_out_of_range_or_non_numeric_coords_are_dropped(self, bad):
        assert parse_click({"x": bad, "y": 0.5, "label": 1}) is None

    def test_booleans_are_not_accepted_as_coordinates(self):
        """bool subclasses int, so True would otherwise sail through as 1."""
        assert parse_click({"x": True, "y": 0.5, "label": 1}) is None

    def test_boolean_labels_are_coerced_rather_than_rejected(self):
        """Known asymmetry, preserved from the original.

        Coordinates reject bools explicitly, but the label check is
        `label not in (0, 1)`, and `True == 1` in Python -- so a model emitting
        `"label": true` gets a foreground click rather than a dropped one.
        Benign in practice (true does mean foreground) but inconsistent; worth
        revisiting once behaviour is allowed to change.
        """
        assert parse_click({"x": 0.5, "y": 0.5, "label": True}) == {
            "x": 0.5, "y": 0.5, "label": 1
        }
        assert parse_click({"x": 0.5, "y": 0.5, "label": False}) == {
            "x": 0.5, "y": 0.5, "label": 0
        }

    @pytest.mark.parametrize("bad", [2, -1, None, "1"])
    def test_invalid_labels_are_dropped(self, bad):
        assert parse_click({"x": 0.5, "y": 0.5, "label": bad}) is None

    def test_cell_coordinates_map_to_the_cell_centre(self):
        assert parse_click({"cell": [0, 0], "label": 1}) == {"x": 0.05, "y": 0.05, "label": 1}
        assert parse_click({"cell": [9, 9], "label": 1}) == {"x": 0.95, "y": 0.95, "label": 1}

    def test_cell_is_row_col_not_x_y(self):
        click = parse_click({"cell": [0, 9], "label": 1})
        assert click == {"x": 0.95, "y": 0.05, "label": 1}

    @pytest.mark.parametrize("cell", [[10, 0], [-1, 0], [0, 10], [1], "nope", [True, 1]])
    def test_bad_cells_are_dropped(self, cell):
        assert parse_click({"cell": cell, "label": 1}) is None

    def test_explicit_xy_wins_over_cell(self):
        click = parse_click({"x": 0.1, "y": 0.2, "cell": [9, 9], "label": 1})
        assert (click["x"], click["y"]) == (0.1, 0.2)

    def test_non_dict_is_dropped(self):
        assert parse_click("nope") is None


class TestSomResponse:
    def test_extracts_accepted_marks(self):
        assert parse_som_response(answer({"accepted_marks": [1, 3, 5]})) == [1, 3, 5]

    def test_booleans_are_not_marks(self):
        assert parse_som_response(answer({"accepted_marks": [True, 2, False]})) == [2]

    def test_non_integers_are_dropped(self):
        assert parse_som_response(answer({"accepted_marks": [1, "2", 3.0, None]})) == [1]

    def test_valid_ids_filters_out_of_range(self):
        text = answer({"accepted_marks": [1, 2, 99]})
        assert parse_som_response(text, valid_ids={1, 2}) == [1, 2]

    def test_the_last_answer_block_wins(self):
        text = answer({"accepted_marks": [1]}) + answer({"accepted_marks": [2]})
        assert parse_som_response(text) == [2]

    @pytest.mark.parametrize("text", MALFORMED)
    def test_malformed_yields_empty(self, text):
        assert parse_som_response(text) == []


class TestCreatureClickGroups:
    def test_parses_a_multi_click_group(self):
        text = answer({"missed_creatures": [
            {"id": 1, "description": "tan crab", "clicks": clicks((0.1, 0.2), (0.15, 0.25))}
        ]})
        (group,) = parse_creature_click_groups(text)
        assert group["id"] == 1
        assert group["description"] == "tan crab"
        assert len(group["clicks"]) == 2

    def test_negative_only_groups_are_dropped(self):
        """Background clicks constrain a target; they cannot seed one."""
        text = answer({"missed_creatures": [
            {"id": 1, "clicks": [{"x": 0.5, "y": 0.5, "label": 0}]}
        ]})
        assert parse_creature_click_groups(text) == []

    def test_a_negative_click_alongside_a_positive_is_kept(self):
        text = answer({"missed_creatures": [{"id": 1, "clicks": [
            {"x": 0.5, "y": 0.5, "label": 0}, {"x": 0.6, "y": 0.6, "label": 1}
        ]}]})
        (group,) = parse_creature_click_groups(text)
        assert [c["label"] for c in group["clicks"]] == [0, 1]

    def test_group_losing_all_clicks_to_validation_is_dropped(self):
        text = answer({"missed_creatures": [{"id": 1, "clicks": [{"x": 5, "y": 5, "label": 1}]}]})
        assert parse_creature_click_groups(text) == []

    def test_missing_ids_are_assigned_the_lowest_free_integers(self):
        text = answer({"missed_creatures": [
            {"clicks": clicks((0.1, 0.1))},
            {"id": 1, "clicks": clicks((0.2, 0.2))},
            {"clicks": clicks((0.3, 0.3))},
        ]})
        assert [g["id"] for g in parse_creature_click_groups(text)] == [2, 1, 3]

    def test_duplicate_ids_are_reassigned(self):
        text = answer({"missed_creatures": [
            {"id": 1, "clicks": clicks((0.1, 0.1))},
            {"id": 1, "clicks": clicks((0.2, 0.2))},
        ]})
        assert [g["id"] for g in parse_creature_click_groups(text)] == [1, 2]

    @pytest.mark.parametrize("bad_id", [0, -1, True, "1", 1.5, None])
    def test_non_positive_int_ids_are_reassigned(self, bad_id):
        text = answer({"missed_creatures": [{"id": bad_id, "clicks": clicks((0.1, 0.1))}]})
        assert parse_creature_click_groups(text)[0]["id"] == 1

    def test_non_string_description_becomes_empty(self):
        text = answer({"missed_creatures": [
            {"id": 1, "description": 123, "clicks": clicks((0.1, 0.1))}
        ]})
        assert parse_creature_click_groups(text)[0]["description"] == ""

    @pytest.mark.parametrize("text", MALFORMED)
    def test_malformed_yields_empty(self, text):
        assert parse_creature_click_groups(text) == []

    def test_wrong_payload_shape_yields_empty(self):
        assert parse_creature_click_groups(answer({"missed_creatures": "nope"})) == []
        assert parse_creature_click_groups(answer({"other_key": []})) == []
        assert parse_creature_click_groups(answer([1, 2])) == []


class TestClickProposals:
    def test_legacy_single_click_contract(self):
        text = answer({"missed_creatures": [{"x": 0.4, "y": 0.6, "description": "eel"}]})
        assert parse_click_proposals(text) == [{"x": 0.4, "y": 0.6, "description": "eel"}]

    def test_out_of_range_dropped_and_description_defaulted(self):
        text = answer({"missed_creatures": [
            {"x": 1.5, "y": 0.5, "description": "gone"},
            {"x": 0.5, "y": 0.5},
        ]})
        assert parse_click_proposals(text) == [{"x": 0.5, "y": 0.5, "description": ""}]


class TestFrameValidity:
    @pytest.mark.parametrize("value", ["usable", "corrupted"])
    def test_reads_the_verdict(self, value):
        assert parse_frame_validity(f"<validity>{value}</validity>") == value

    def test_is_case_insensitive(self):
        assert parse_frame_validity("<validity>CORRUPTED</validity>") == "corrupted"

    def test_last_verdict_wins(self):
        text = "<validity>usable</validity><validity>corrupted</validity>"
        assert parse_frame_validity(text) == "corrupted"

    @pytest.mark.parametrize("text", ["", "nothing", None, "<validity>maybe</validity>"])
    def test_missing_or_unknown_is_none(self, text):
        assert parse_frame_validity(text) is None


class TestRefinement:
    @pytest.mark.parametrize("action", ["accept", "reject"])
    def test_terminal_actions_carry_no_points(self, action):
        text = f"<refine>{json.dumps({'action': action})}</refine>"
        assert parse_refinement_response(text) == {"action": action, "add_points": []}

    def test_refine_collects_valid_points(self):
        payload = {"action": "refine", "add_points": [
            {"x": 0.1, "y": 0.1, "label": 1},
            {"x": 9.9, "y": 0.1, "label": 1},
        ]}
        result = parse_refinement_response(f"<refine>{json.dumps(payload)}</refine>")
        assert result == {"action": "refine", "add_points": [{"x": 0.1, "y": 0.1, "label": 1}]}

    def test_refine_without_add_points_is_malformed(self):
        """Asking to refine while supplying nothing to refine with."""
        assert parse_refinement_response('<refine>{"action": "refine"}</refine>') is None

    def test_unknown_action_is_none(self):
        assert parse_refinement_response('<refine>{"action": "explode"}</refine>') is None

    @pytest.mark.parametrize("text", ["", "no tag", None, "<refine>bad</refine>"])
    def test_malformed_is_none(self, text):
        assert parse_refinement_response(text) is None


class TestClickRefinement:
    @pytest.mark.parametrize("action", ["ok", "move", "drop"])
    def test_actions(self, action):
        text = f"<click_refine>{json.dumps({'action': action})}</click_refine>"
        assert parse_click_refinement_response(text)["action"] == action

    def test_move_carries_validated_clicks(self):
        payload = {"action": "move", "new_clicks": [
            {"x": 0.3, "y": 0.3, "label": 1}, {"x": 2.0, "y": 0.3, "label": 1}
        ]}
        text = f"<click_refine>{json.dumps(payload)}</click_refine>"
        result = parse_click_refinement_response(text)
        assert result["new_clicks"] == [{"x": 0.3, "y": 0.3, "label": 1}]

    def test_unknown_action_is_none(self):
        text = '<click_refine>{"action": "x"}</click_refine>'
        assert parse_click_refinement_response(text) is None
