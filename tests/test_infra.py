"""The infra scripts must carry no host details and must not delete remote state."""
from __future__ import annotations

from pathlib import Path

import pytest
from hostcheck import find as find_host_detail

INFRA = Path(__file__).parent.parent / "infra"
SCRIPTS = sorted(INFRA.glob("*.sh"))


def test_there_are_infra_scripts():
    assert SCRIPTS, "expected the remote helpers under infra/"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_hostname_username_or_absolute_path_is_baked_in(path):
    """These were untracked in the old repo precisely because they were not."""
    text = path.read_text()
    hits = find_host_detail(text)
    assert not hits, f"{path.name} carries machine-specific detail: {hits}"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_required_environment_is_asserted_not_defaulted(path):
    """A silent default would sync to, or run on, the wrong machine."""
    text = path.read_text()
    assert 'MAL_REMOTE_HOST:?' in text
    assert 'MAL_REMOTE_ROOT:?' in text


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_scripts_fail_fast(path):
    assert "set -euo pipefail" in path.read_text()


class TestSyncSafety:
    SYNC = INFRA / "sync_to_remote.sh"

    def test_never_deletes_remote_files(self):
        """Remote run outputs and datasets are persistent state; a SOURCE sync
        must not remove them.

        Checked on executable lines only -- the script's own comment explains
        that --delete is deliberately absent, and matching that comment would
        make this test pass for the wrong reason.
        """
        code = [
            line for line in self.SYNC.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        offenders = [line for line in code if "--delete" in line]
        assert not offenders, f"--delete used in: {offenders}"

    @pytest.mark.parametrize(
        "excluded",
        [".env", "runs/", "outputs/", "weights/", "checkpoints/", "assets/videos/",
         "*.pth", "*.safetensors", "*.mp4", ".venv/", "docs/LOCAL_NOTES.md",
         "docs/*_assets/"],
    )
    def test_excludes_secrets_data_and_outputs(self, excluded):
        assert f"--exclude='{excluded}'" in self.SYNC.read_text()

    def test_defaults_to_a_dry_run(self):
        text = self.SYNC.read_text()
        assert "--apply" in text
        assert "rsync_args+=(--dry-run)" in text
