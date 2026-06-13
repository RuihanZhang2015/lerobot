#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

POLICY_PATH="${POLICY_PATH:-outputs/train/act_so101_pick_place/checkpoints/020000/pretrained_model}"
DURATION="${DURATION:-10}"

require_paths "$POLICY_PATH/model.safetensors" "$FOLLOWER_PORT" "$CAMERA"

echo "Starting autonomous rollout for ${DURATION}s. Keep emergency power access ready."

"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  lerobot-rollout \
  --strategy.type=base \
  "--policy.path=$POLICY_PATH" \
  --policy.device=cuda \
  --robot.type=so101_follower \
  "--robot.port=$FOLLOWER_PORT" \
  --robot.id=my_so101_follower \
  "--robot.cameras={front: {type: opencv, index_or_path: $CAMERA, width: 640, height: 480, fps: 30}}" \
  "--task=$TASK" \
  --fps=30 \
  "--duration=$DURATION" \
  --display_data=true \
  --return_to_initial_position=false
