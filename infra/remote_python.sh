#!/usr/bin/env bash
# Run a script on the GPU host's virtualenv.
#
#   infra/remote_python.sh scripts/whatever.py --flag
#
# Host and paths come from the environment so this file carries none:
#   MAL_REMOTE_HOST     ssh alias or user@host      (required)
#   MAL_REMOTE_ROOT     repo path on that host      (required)
#   MAL_REMOTE_PYTHON   interpreter, relative or absolute (default .venv/bin/python)
#
# Put them in .env, or export them. See .env.example.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <script-or-python-args> [...]" >&2
  exit 2
fi

: "${MAL_REMOTE_HOST:?set MAL_REMOTE_HOST (ssh alias or user@host)}"
: "${MAL_REMOTE_ROOT:?set MAL_REMOTE_ROOT (repo path on the remote host)}"
remote_python="${MAL_REMOTE_PYTHON:-.venv/bin/python}"

printf -v remote_args '%q ' "$@"
exec ssh -t "$MAL_REMOTE_HOST" \
  "cd ${MAL_REMOTE_ROOT} && PYTHONPATH=. ${remote_python} ${remote_args}"
