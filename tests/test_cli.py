"""CLI argument handling and config loading."""
from __future__ import annotations

import json

import pytest

from marine_autolabel.cli.run_pipeline import build_parser, main
from marine_autolabel.config import RunConfig


@pytest.fixture
def run_config(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"frames": [{"id": "a"}, {"id": "b"}]}))
    path = tmp_path / "run.json"
    path.write_text(json.dumps({
        "benchmark_id": "test",
        "output_dir": str(tmp_path / "out"),
        "manifest": str(manifest),
        "models": {"click": "claude-opus-5", "phrase": ""},
        "clicks": {"temporal_offsets": [5, 10], "max_repair_rounds": 2},
    }))
    return path


class TestConfigLoading:
    def test_loads_a_config_file(self, run_config):
        config = RunConfig.from_file(run_config)
        assert config.benchmark_id == "test"
        assert config.models.click == "claude-opus-5"
        assert config.clicks.max_repair_rounds == 2

    def test_json_lists_become_tuples(self, run_config):
        assert RunConfig.from_file(run_config).clicks.temporal_offsets == (5, 10)

    def test_a_written_config_round_trips(self, tmp_path, run_config):
        original = RunConfig.from_file(run_config)
        out = tmp_path / "written.json"
        original.write(out)
        assert RunConfig.from_file(out).to_dict() == original.to_dict()

    def test_a_typo_is_rejected_rather_than_ignored(self, tmp_path):
        """Otherwise the default silently stands and the run is wrong."""
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "benchmark_id": "x", "output_dir": "/tmp/o", "manifest": "m.json",
            "mask_generatr": "hybrid",
        }))
        with pytest.raises(ValueError, match="unknown config key"):
            RunConfig.from_file(path)

    def test_a_typo_in_a_nested_section_is_rejected(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "benchmark_id": "x", "output_dir": "/tmp/o", "manifest": "m.json",
            "models": {"clik": "claude-opus-5"},
        }))
        with pytest.raises(ValueError, match="unknown models key"):
            RunConfig.from_file(path)

    def test_the_error_lists_the_valid_keys(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "benchmark_id": "x", "output_dir": "/t", "manifest": "m", "clicks": {"stratgy": "s"},
        }))
        with pytest.raises(ValueError, match="strategy"):
            RunConfig.from_file(path)


class TestParser:
    def test_config_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_frame_id_is_repeatable(self):
        args = build_parser().parse_args(
            ["--config", "c.json", "--frame-id", "a", "--frame-id", "b"]
        )
        assert args.frame_ids == ["a", "b"]

    def test_resume_is_the_default(self):
        assert build_parser().parse_args(["--config", "c.json"]).no_resume is False


class TestDryRun:
    def test_reports_what_would_run_without_touching_sam3(self, run_config, capsys):
        assert main(["--config", str(run_config), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "would run   : ['a', 'b']" in out
        assert "claude-opus-5" in out

    def test_frame_id_narrows_the_selection(self, run_config, capsys):
        main(["--config", str(run_config), "--frame-id", "b", "--dry-run"])
        assert "would run   : ['b']" in capsys.readouterr().out

    def test_an_empty_phrase_model_is_reported_as_the_click_model(self, run_config, capsys):
        main(["--config", str(run_config), "--dry-run"])
        assert "phrase=claude-opus-5" in capsys.readouterr().out
