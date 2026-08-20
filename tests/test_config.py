"""RunConfig must absorb the run summaries the old pipeline already emits.

This is the phase-1 guard rail: if a field is dropped or renamed during the
migration, an existing run directory stops round-tripping and this fails.
"""
from __future__ import annotations

from pathlib import Path

from marine_autolabel.config import ClickEngineConfig, ModelRoles, RunConfig, TextProposalConfig


def _flatten(d: dict, out: set[str]) -> set[str]:
    for k, v in d.items():
        out.add(k)
        if isinstance(v, dict):
            _flatten(v, out)
    return out


def test_covers_every_key_in_a_real_run_summary(run_summary):
    cfg = RunConfig(
        benchmark_id="x", output_dir=Path("/tmp"), manifest=Path("m.json")
    ).to_dict()
    ours = _flatten(cfg, set())

    # Keys the summary records that are results or provenance, not configuration.
    not_config = {
        "frames",
        "anthropic_effort",
        "firstpass_model",
        "source_provenance",
        "api_failure_count",
    }
    theirs = set(run_summary) - not_config

    aliases = {
        "model": "click",
        "phrase_model": "phrase",
        "whole_frame_review": "whole_frame_review",
        "text_proposal_mode": "mode",
        "text_proposal_threshold": "threshold",
        "min_text_confidence": "min_confidence",
    }
    missing = {k for k in theirs if aliases.get(k, k) not in ours}
    assert not missing, f"RunConfig does not cover: {sorted(missing)}"


def test_defaults_mirror_the_most_recent_scored_run():
    """Model choice is NOT settled; these track the latest configuration."""
    m = ModelRoles()
    assert m.click == m.verify == "claude-opus-5"
    assert m.firstpass == m.quality_screen == "claude-sonnet-5"
    assert m.effort == "medium"


def test_an_empty_phrase_model_falls_back_to_the_click_model():
    assert ModelRoles(phrase="").resolved_phrase() == ModelRoles().click


def test_structural_split_holds_whatever_the_models():
    """The stronger model clicks and verifies; a cheaper one does first pass."""
    m = ModelRoles()
    assert m.click == m.verify
    assert m.firstpass == m.quality_screen


def test_pipeline_defaults_match_the_settled_findings():
    """Unlike the model choice, these ARE settled. See docs/findings.md."""
    c, t = ClickEngineConfig(), TextProposalConfig()
    # The whole-frame MLLM review pass cost recall; it must stay off.
    assert c.whole_frame_review is False
    assert c.mask_generator == "hybrid"
    assert c.strategy == "S3_cons_temp"
    assert t.mode == "adaptive"


def test_config_serialises_paths_and_tuples(tmp_path):
    cfg = RunConfig(
        benchmark_id="x", output_dir=tmp_path, manifest=Path("m.json"), frame_ids=("a", "b")
    )
    d = cfg.to_dict()
    assert d["output_dir"] == str(tmp_path)
    assert d["frame_ids"] == ["a", "b"]
    assert d["clicks"]["temporal_offsets"] == [15, 30, 45]

    out = tmp_path / "run_config.json"
    cfg.write(out)
    payload = __import__("json").loads(out.read_text())
    assert "config" in payload
    assert "git" in payload["source_provenance"]
