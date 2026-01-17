#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
EXTRAS="${EXTRAS:-hf}"
NO_DEV="${NO_DEV:-0}"
CTOK_DEFAULT_TASK="${CTOK_DEFAULT_TASK:-waf_http}"
CTOK_DEFAULT_MODEL="${CTOK_DEFAULT_MODEL:-roberta_base}"
CTOK_DEFAULT_TOKENIZER="${CTOK_DEFAULT_TOKENIZER:-ctok_v1}"
CTOK_CONFIG_ROOT="${CTOK_CONFIG_ROOT:-$ROOT_DIR}"
CTOK_ARTIFACTS_DIR="${CTOK_ARTIFACTS_DIR:-$ROOT_DIR/artifacts/tokenizers}"
CTOK_RESULTS_DIR="${CTOK_RESULTS_DIR:-$ROOT_DIR/results/runs}"
CTOK_ENV_FILE="${CTOK_ENV_FILE:-$ROOT_DIR/.ctok_env}"

usage() {
  cat <<'EOF'
Usage: ./setup.sh [--extras <list>] [--no-extras] [--no-dev]

Creates a virtual environment with uv and syncs dependencies from pyproject.toml.
Examples:
  ./setup.sh
  ./setup.sh --extras hf,transformers
  EXTRAS=hf ./setup.sh
  VENV_DIR=/tmp/ctok-venv ./setup.sh
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --extras)
      if [ $# -lt 2 ] || [ -z "${2:-}" ]; then
        echo "Missing value for --extras" >&2
        exit 1
      fi
      EXTRAS="$2"
      shift 2
      ;;
    --no-extras)
      EXTRAS=""
      shift
      ;;
    --no-dev)
      NO_DEV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it from https://github.com/astral-sh/uv" >&2
  exit 1
fi

uv venv "$VENV_DIR"

SYNC_ARGS=()
if [ "$NO_DEV" -eq 1 ]; then
  SYNC_ARGS+=(--no-dev)
fi
if [ -n "$EXTRAS" ]; then
  IFS=',' read -r -a extra_list <<<"$EXTRAS"
  for extra in "${extra_list[@]}"; do
    extra="${extra//[[:space:]]/}"
    if [ -n "$extra" ]; then
      SYNC_ARGS+=(--extra "$extra")
    fi
  done
fi

VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$PATH" uv sync --active "${SYNC_ARGS[@]}"

if [ -n "$CTOK_ENV_FILE" ]; then
  cat > "$CTOK_ENV_FILE" <<EOF
export CTOK_DEFAULT_TASK="$CTOK_DEFAULT_TASK"
export CTOK_DEFAULT_MODEL="$CTOK_DEFAULT_MODEL"
export CTOK_DEFAULT_TOKENIZER="$CTOK_DEFAULT_TOKENIZER"
export CTOK_CONFIG_ROOT="$CTOK_CONFIG_ROOT"
export CTOK_ARTIFACTS_DIR="$CTOK_ARTIFACTS_DIR"
export CTOK_RESULTS_DIR="$CTOK_RESULTS_DIR"
EOF
fi

cat <<EOF
Setup complete.
Activate the environment with:
  source "$VENV_DIR/bin/activate"
Load ctok defaults with:
  source "$CTOK_ENV_FILE"
EOF
