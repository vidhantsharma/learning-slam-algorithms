#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
VENV_DIR="$ROOT_DIR/.venv"

usage() {
  cat <<EOF
Usage: ./run_demo.sh [--algo ALGO] [--animate] [--show] [--steps N]

Options:
  --algo ALGO  Algorithm to run: ekf_slam | fast_slam | grid_slam  (default: ekf_slam)
  --animate    Run animated visualization (default: static plot)
  --show       Show the plot window (default: save image only)
  --steps N    Number of timesteps in the demo (default: 250)
EOF
}

ALGO="ekf_slam"
ANIMATE=false
SHOW=false
STEPS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo)
      ALGO="$2"
      shift 2
      ;;
    --animate)
      ANIMATE=true
      shift
      ;;
    --show)
      SHOW=true
      shift
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# ── Validate --algo ──────────────────────────────────────────────────────────
case "$ALGO" in
  ekf_slam|fast_slam|grid_slam)
    ;;
  *)
    echo "Error: unknown algorithm '$ALGO'. Choose ekf_slam, fast_slam, or grid_slam."
    usage
    exit 1
    ;;
esac

# ── Build ────────────────────────────────────────────────────────────────────
mkdir -p "$BUILD_DIR"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR"

# ── Run C++ demo ─────────────────────────────────────────────────────────────
DEMO_CMD=("$BUILD_DIR/examples/${ALGO}_demo")
if [[ -n "$STEPS" ]]; then
  DEMO_CMD+=(--steps "$STEPS")
fi
"${DEMO_CMD[@]}"

# ── Python virtual environment ───────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

REQ_FILE="$ROOT_DIR/requirements.txt"
REQ_HASH_FILE="$VENV_DIR/.requirements.sha"
CUR_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"

if [[ ! -f "$REQ_HASH_FILE" ]] || [[ "$(cat "$REQ_HASH_FILE")" != "$CUR_HASH" ]]; then
  pip install -r "$REQ_FILE"
  echo "$CUR_HASH" > "$REQ_HASH_FILE"
fi

# ── Run visualization ────────────────────────────────────────────────────────
PLOT_CMD=(python3 "$ROOT_DIR/slam_viz/plot_${ALGO}.py")
if [[ "$ANIMATE" == "true" ]]; then
  PLOT_CMD+=(--animate)
fi
if [[ "$SHOW" == "true" ]]; then
  PLOT_CMD+=(--show)
fi
"${PLOT_CMD[@]}"
