"""The wiring module's environment guards and path resolution.

The wiring itself needs CUDA and a key, but its failure modes must be clear
without either -- a run that dies on a missing extra should say so, not
traceback from an import three modules deep.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from marine_autolabel.cli._wiring import MissingExtraError, resolve_path


class TestPathResolution:
    def test_a_plain_path_passes_through(self):
        assert resolve_path("/data/videos/clip.mp4") == Path("/data/videos/clip.mp4")

    def test_a_placeholder_is_expanded_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("MAL_SEATUBE_ROOT", "/data/seatube")
        resolved = resolve_path("${MAL_SEATUBE_ROOT}/clip/clip_10s.mp4")
        assert resolved == Path("/data/seatube/clip/clip_10s.mp4")

    def test_an_explicit_root_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("MAL_SEATUBE_ROOT", "/from/env")
        resolved = resolve_path("${MAL_SEATUBE_ROOT}/x", {"MAL_SEATUBE_ROOT": "/explicit"})
        assert resolved == Path("/explicit/x")

    def test_an_unset_root_names_itself_and_points_at_the_fix(self, monkeypatch):
        monkeypatch.delenv("MAL_SEATUBE_ROOT", raising=False)
        with pytest.raises(MissingExtraError) as excinfo:
            resolve_path("${MAL_SEATUBE_ROOT}/clip.mp4")
        message = str(excinfo.value)
        assert "MAL_SEATUBE_ROOT" in message
        assert ".env.example" in message

    def test_multiple_placeholders_resolve(self, monkeypatch):
        monkeypatch.setenv("MAL_RUNS_ROOT", "/runs")
        assert resolve_path("${MAL_RUNS_ROOT}/a/b") == Path("/runs/a/b")

    def test_a_home_shortcut_is_expanded(self):
        assert not str(resolve_path("~/clip.mp4")).startswith("~")


class TestEnvironmentGuards:
    def test_a_missing_api_key_is_reported_clearly(self, monkeypatch):
        from marine_autolabel.cli import _wiring

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(MissingExtraError, match="ANTHROPIC_API_KEY"):
            _wiring._require_api_key()

    def test_a_present_api_key_passes(self, monkeypatch):
        from marine_autolabel.cli import _wiring

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        _wiring._require_api_key()

    def test_a_missing_sam3_extra_names_the_install_command(self):
        """The guard must check the sam3 package itself.

        sam3svc.service imports cleanly without it -- build_sam3_service defers
        its sam3 import to call time -- so guarding on the wrapper would pass
        here and fail later, deep inside a run.
        """
        import importlib.util

        from marine_autolabel.cli import _wiring

        if importlib.util.find_spec("sam3") is not None:  # pragma: no cover
            pytest.skip("sam3 is installed in this environment")

        with pytest.raises(MissingExtraError) as excinfo:
            _wiring._require_sam3()
        assert "pip install -e '.[dev,sam3]'" in str(excinfo.value)
        assert "sam3" in str(excinfo.value)

    def test_the_wrapper_alone_would_not_have_caught_it(self):
        """Documents why the guard checks the package, not the import."""
        from marine_autolabel.sam3svc.service import build_sam3_service

        assert callable(build_sam3_service), "imports fine even without sam3 present"
