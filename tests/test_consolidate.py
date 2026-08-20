"""Consolidating masks judged to be one organism."""
from __future__ import annotations

import numpy as np

from marine_autolabel.postprocess.consolidate import consolidate_masks

H = W = 200


def entry(y0, y1, x0, x1, **extra):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return {"mask": m, **extra}


ADJACENT = [entry(20, 60, 20, 60), entry(20, 60, 60, 100)]   # touching -> nominated
DISTANT = [entry(10, 40, 10, 40), entry(150, 190, 150, 190)]  # never nominated


def merge_all(candidates, components):
    return components


def merge_none(candidates, components):
    return []


class TestNoNomination:
    def test_distant_masks_are_left_alone(self):
        out, audit = consolidate_masks(
            DISTANT, partition=merge_all, verify_union=lambda r, g: True
        )
        assert len(out) == 2
        assert audit["components"] == []
        assert audit["n_consolidated_components"] == 0

    def test_the_partition_is_not_even_consulted(self):
        calls = []
        consolidate_masks(
            DISTANT,
            partition=lambda c, comp: calls.append(comp) or [],
            verify_union=lambda r, g: True,
        )
        assert calls == []


class TestBothGates:
    def test_a_merge_needs_partition_and_verification(self):
        out, audit = consolidate_masks(
            ADJACENT, partition=merge_all, verify_union=lambda r, g: True
        )
        assert len(out) == 1
        assert audit["n_consolidated_components"] == 1
        assert audit["n_masks_removed"] == 1
        assert out[0]["consolidated_from"] == [0, 1]

    def test_the_partition_gate_alone_can_refuse(self):
        """Geometry nominated them; identity review said they are distinct."""
        out, audit = consolidate_masks(
            ADJACENT, partition=merge_none, verify_union=lambda r, g: True
        )
        assert len(out) == 2
        assert audit["same_identity_groups"] == []
        assert audit["union_attempts"][0]["reason"] == "no_same_identity"

    def test_the_union_verifier_alone_can_refuse(self):
        """The merged mask must clear the same bar as any accepted mask."""
        out, audit = consolidate_masks(
            ADJACENT, partition=merge_all, verify_union=lambda r, g: False
        )
        assert len(out) == 2
        assert audit["union_attempts"][0]["union_verified"] is False
        assert audit["n_masks_removed"] == 0

    def test_a_failed_merge_preserves_the_originals_exactly(self):
        out, _ = consolidate_masks(
            ADJACENT, partition=merge_all, verify_union=lambda r, g: False
        )
        assert np.array_equal(out[0]["mask"], ADJACENT[0]["mask"])
        assert np.array_equal(out[1]["mask"], ADJACENT[1]["mask"])


class TestUnionContent:
    def test_the_union_covers_both_members(self):
        seen = {}

        def verify(result, subgroup):
            seen["mask"] = result["mask"]
            seen["subgroup"] = subgroup
            return True

        consolidate_masks(ADJACENT, partition=merge_all, verify_union=verify)
        assert seen["subgroup"] == [0, 1]
        assert seen["mask"][30, 30] and seen["mask"][30, 80]

    def test_the_anchor_is_the_lowest_index_and_keeps_its_metadata(self):
        masks = [entry(20, 60, 20, 60, source="firstpass", prob=0.9),
                 entry(20, 60, 60, 100, source="som")]
        out, _ = consolidate_masks(
            masks, partition=merge_all, verify_union=lambda r, g: True
        )
        assert out[0]["prob"] == 0.9, "anchor metadata survives"
        assert out[0]["source"] == "overlap_identity_union"

    def test_uninvolved_masks_keep_their_positions(self):
        masks = [entry(150, 190, 150, 190), *ADJACENT]
        out, _ = consolidate_masks(
            masks,
            partition=lambda c, comp: comp,
            verify_union=lambda r, g: True,
        )
        assert len(out) == 2
        assert np.array_equal(out[0]["mask"], masks[0]["mask"]), "the distant mask is first"

    def test_a_three_way_component_merges_into_one(self):
        chain = [entry(20, 60, 20, 60), entry(20, 60, 60, 100), entry(20, 60, 100, 140)]
        out, audit = consolidate_masks(
            chain, partition=merge_all, verify_union=lambda r, g: True
        )
        assert len(out) == 1
        assert audit["n_masks_removed"] == 2
        assert out[0]["consolidated_from"] == [0, 1, 2]


class TestPartitionFreedom:
    def test_a_component_may_split_into_a_smaller_subgroup(self):
        """Geometry can chain distinct organisms; the partition may keep only some."""
        chain = [entry(20, 60, 20, 60), entry(20, 60, 60, 100), entry(20, 60, 100, 140)]
        out, audit = consolidate_masks(
            chain, partition=lambda c, comp: [[0, 1]], verify_union=lambda r, g: True
        )
        assert len(out) == 2
        assert audit["same_identity_groups"] == [[0, 1]]

    def test_single_member_subgroups_are_ignored(self):
        out, _ = consolidate_masks(
            ADJACENT, partition=lambda c, comp: [[0], [1]], verify_union=lambda r, g: True
        )
        assert len(out) == 2

    def test_the_audit_records_the_geometry_that_triggered_review(self):
        _, audit = consolidate_masks(
            ADJACENT, partition=merge_none, verify_union=lambda r, g: True
        )
        assert audit["components"] == [[0, 1]]
        assert audit["candidate_pairs"][0]["trigger"] in {"overlap", "near_contact"}


class TestEmptyInput:
    def test_no_masks(self):
        out, audit = consolidate_masks(
            [], partition=merge_all, verify_union=lambda r, g: True
        )
        assert out == [] and audit["components"] == []

    def test_one_mask(self):
        out, _ = consolidate_masks(
            [entry(10, 40, 10, 40)], partition=merge_all, verify_union=lambda r, g: True
        )
        assert len(out) == 1
