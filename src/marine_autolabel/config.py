"""Run configuration.

Replaces the ~40 CLI flags on the old `run_presentation_custom_flow.py`. Field
names and defaults are taken from the `summary.json` that runs already emit, so
an existing run directory round-trips into a RunConfig unchanged.

A run is fully described by (RunConfig, git SHA). Both are written into the
output directory at start, which makes the "record the commit SHA with the run"
rule in the old AGENTS.md automatic instead of manual.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TextProposalMode = Literal["none", "adaptive", "compact", "broad"]
FinderMode = Literal["mask-guided", "hybrid", "s3"]
MaskGenerator = Literal["hybrid", "mm", "mmzoom"]
BorderScan = Literal["never", "last", "always"]
Profile = Literal["general", "underwater"]


@dataclass(frozen=True)
class ModelRoles:
    """Which model drives which stage.

These defaults are a starting configuration, not a concluded experiment --
    which model belongs in which role is an empirical question for your data.

    The structural split is the durable part: the stronger vision model clicks
    and verifies, a cheaper one handles first-pass text prompting and the
    frame-quality screen.
    """

    click: str = "claude-opus-5"
    verify: str = "claude-opus-5"
    firstpass: str = "claude-sonnet-5"
    quality_screen: str = "claude-sonnet-5"
    phrase: str = "claude-sonnet-5"  # empty -> use `click`
    effort: str = "medium"  # "" | low | medium | high | xhigh | max

    def resolved_phrase(self) -> str:
        return self.phrase or self.click


@dataclass(frozen=True)
class ClickEngineConfig:
    strategy: str = "S3_cons_temp"
    finder_mode: FinderMode = "mask-guided"
    mask_generator: MaskGenerator = "hybrid"
    whole_frame_review: bool = False  # do not enable; it costs recall
    max_click_groups_per_pass: int = 0  # 0 = unbounded
    mask_guided_passes: int = 0  # 0 = adaptive
    sparse_extra_pass: bool = False
    border_scan: BorderScan = "last"
    zoom_crop_frac: float = 0.50
    click_localization_crop_frac: float = 0.65
    min_recovery_confidence: float = 0.50
    max_repair_rounds: int = 4  # bounds the post-verify repair loop; 0 disables repair
    persistent_firstpass_mask_guidance: bool = True
    persistent_click_masks_between_passes: bool = True
    temporal_offsets: tuple[int, ...] = (15, 30, 45)
    temporal_seconds: tuple[float, ...] = (0.5, 1.0)
    click_dedup_px: int = 40
    mask_nms_iou: float = 0.70


@dataclass(frozen=True)
class TextProposalConfig:
    mode: TextProposalMode = "adaptive"
    threshold: float = 0.40
    min_confidence: float = 0.50


@dataclass(frozen=True)
class RunConfig:
    benchmark_id: str
    output_dir: Path
    manifest: Path
    profile: Profile = "underwater"
    models: ModelRoles = field(default_factory=ModelRoles)
    clicks: ClickEngineConfig = field(default_factory=ClickEngineConfig)
    text: TextProposalConfig = field(default_factory=TextProposalConfig)

    firstpass_root: Path | None = None
    firstpass_skipped: bool = False
    initial_mask_root: Path | None = None
    selection_manifest: Path | None = None
    frame_ids: tuple[str, ...] = ()  # empty = all frames in the manifest
    visual_qa_focus: str | None = None
    exploratory_not_scored: bool = False  # true => excluded from scored results

    def to_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, (tuple, list)):
                return [encode(v) for v in value]
            if isinstance(value, dict):
                # dataclasses.asdict() has already flattened nested dataclasses
                # into dicts, but leaves tuples and Paths inside them untouched.
                return {k: encode(v) for k, v in value.items()}
            return value

        return {k: encode(v) for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunConfig:
        """Build a RunConfig from a plain dict, e.g. a loaded JSON file.

        Unknown keys are rejected rather than ignored: a typo in a run config
        would otherwise silently leave the default in place, and these runs are
        too expensive to discover that from the results.
        """
        payload = dict(payload)
        nested = {
            "models": ModelRoles,
            "clicks": ClickEngineConfig,
            "text": TextProposalConfig,
        }
        built: dict[str, Any] = {}
        for key, klass in nested.items():
            section = payload.pop(key, None)
            if section is not None:
                _reject_unknown(section, klass, key)
                built[key] = klass(**_coerce_tuples(section, klass))

        _reject_unknown(payload, cls, "config")
        for field_name in ("output_dir", "manifest", "firstpass_root",
                           "initial_mask_root", "selection_manifest"):
            if isinstance(payload.get(field_name), str):
                payload[field_name] = Path(payload[field_name])
        if isinstance(payload.get("frame_ids"), list):
            payload["frame_ids"] = tuple(payload["frame_ids"])
        return cls(**payload, **built)

    @classmethod
    def from_file(cls, path: Path) -> RunConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        # Accept both a bare config and the {"config": ..., "source_provenance": ...}
        # shape that write() emits, so a run directory round-trips.
        return cls.from_dict(payload.get("config", payload))

    def write(self, path: Path, sources: list[Path] | None = None) -> None:
        payload = {
            "config": self.to_dict(),
            "source_provenance": source_provenance(sources),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _field_names(klass: Any) -> set[str]:
    return {f.name for f in dataclasses.fields(klass)}


def _reject_unknown(section: dict[str, Any], klass: Any, label: str) -> None:
    unknown = set(section) - _field_names(klass)
    if unknown:
        known = sorted(_field_names(klass))
        raise ValueError(f"unknown {label} key(s) {sorted(unknown)}; expected one of {known}")


def _coerce_tuples(section: dict[str, Any], klass: Any) -> dict[str, Any]:
    """JSON has no tuples; restore them for fields declared as such."""
    out = dict(section)
    for field_def in dataclasses.fields(klass):
        value = out.get(field_def.name)
        if isinstance(value, list) and "tuple" in str(field_def.type):
            out[field_def.name] = tuple(value)
    return out


def source_provenance(sources: list[Path] | None = None) -> dict[str, Any]:
    """Git state plus per-file digests, stamped into every run directory.

    The git SHA alone is not enough: scored runs have historically been made from
    dirty trees, so the per-file sha256 is what actually identifies the code that
    produced a result. Mirrors the `source_provenance` block the old pipeline
    already writes into summary.json.
    """
    return {
        "git": git_state(),
        "sha256": {str(p): sha256_file(p) for p in (sources or []) if p.is_file()},
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo: Path | None = None) -> dict[str, Any]:
    """Commit SHA plus a dirty flag, stamped into every run directory."""
    cwd = str(repo) if repo else None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "dirty": None}
