#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

DATASET_ROOT="${DATASET_ROOT:-$HOME/.cache/huggingface/lerobot/ruihanzhang2015/so101-pick-and-place_20260612_174844}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/act_so101_pick_place}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_FREQ="${SAVE_FREQ:-5000}"

require_paths "$DATASET_ROOT/meta/info.json"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  echo "Set OUTPUT_DIR to a new path before starting another training run." >&2
  exit 2
fi

mkdir -p "$(dirname -- "$OUTPUT_DIR")" "$(dirname -- "$LOG_FILE")"

"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  lerobot-train \
  "--dataset.repo_id=$REPO_ID" \
  "--dataset.root=$DATASET_ROOT" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  "--output_dir=$OUTPUT_DIR" \
  --job_name=act_so101_pick_place \
  "--batch_size=$BATCH_SIZE" \
  "--steps=$STEPS" \
  --num_workers=8 \
  --log_freq=100 \
  "--save_freq=$SAVE_FREQ" \
  --wandb.enable=false 2>&1 | tee "$LOG_FILE"
