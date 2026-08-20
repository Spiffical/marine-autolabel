#!/usr/bin/env bash
# Sync SOURCE ONLY to the GPU host. Never data, weights, secrets or outputs.
#
#   infra/sync_to_remote.sh            preview (dry run)
#   infra/sync_to_remote.sh --apply    transfer
#
# Environment (see .env.example):
#   MAL_REMOTE_HOST   ssh alias or user@host   (required)
#   MAL_REMOTE_ROOT   repo path on that host   (required)
#
# Deliberately NOT --delete: remote run outputs and datasets are persistent
# state that a source sync must never remove.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

apply_sync=0
case "${1:-}" in
  --apply) apply_sync=1 ;;
  "")      ;;
  *)       echo "Usage: $0 [--apply]" >&2; exit 2 ;;
esac

: "${MAL_REMOTE_HOST:?set MAL_REMOTE_HOST (ssh alias or user@host)}"
: "${MAL_REMOTE_ROOT:?set MAL_REMOTE_ROOT (repo path on the remote host)}"

cd "$repo_root"

rsync_args=(-a --checksum --no-times --itemize-changes)
[[ $apply_sync -eq 0 ]] && rsync_args+=(--dry-run)

rsync "${rsync_args[@]}" \
  --exclude='/.git' \
  --exclude='.env' \
  --exclude='.DS_Store' \
  --exclude='.venv/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.egg-info/' \
  --exclude='docs/LOCAL_NOTES.md' \
  --exclude='docs/*_assets/' \
  --exclude='runs/' \
  --exclude='outputs/' \
  --exclude='assets/videos/' \
  --exclude='weights/' \
  --exclude='checkpoints/' \
  --exclude='*.mp4' --exclude='*.mov' --exclude='*.avi' --exclude='*.mkv' \
  --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.safetensors' \
  ./ "${MAL_REMOTE_HOST}:${MAL_REMOTE_ROOT}/"

if [[ $apply_sync -eq 0 ]]; then
  echo
  echo 'Dry run only. Re-run with --apply to transfer these source changes.'
fi
