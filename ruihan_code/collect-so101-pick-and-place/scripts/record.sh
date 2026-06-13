#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

EPISODES=50
ROOT=""
RESUME=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --root)
      ROOT="$2"
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$RESUME" == true && -z "$ROOT" ]]; then
  echo "--resume requires --root pointing to the exact existing dataset." >&2
  exit 2
fi

require_paths "$FOLLOWER_PORT" "$LEADER_PORT" "$CAMERA"

ARGS=(
  lerobot-record
  --robot.type=so101_follower
  "--robot.port=$FOLLOWER_PORT"
  --robot.id=my_so101_follower
  "--robot.cameras={front: {type: opencv, index_or_path: $CAMERA, width: 640, height: 480, fps: 30}}"
  --teleop.type=so101_leader
  "--teleop.port=$LEADER_PORT"
  --teleop.id=my_so101_leader
  "--dataset.repo_id=$REPO_ID"
  "--dataset.single_task=$TASK"
  "--dataset.num_episodes=$EPISODES"
  --dataset.episode_time_s=10
  --dataset.reset_time_s=5
  --dataset.push_to_hub=false
  --dataset.streaming_encoding=true
  --dataset.encoder_threads=2
  --display_data=true
)

if [[ -n "$ROOT" ]]; then
  ARGS+=("--dataset.root=$ROOT")
fi

if [[ "$RESUME" == true ]]; then
  ARGS+=(--resume=true)
fi

echo "Starting $EPISODES episode(s). Use Escape to stop and finalize cleanly."
PYNPUT_BACKEND=xorg DISPLAY="${DISPLAY:-:0}" \
  "$CONDA_BIN" run -n "$CONDA_ENV" "${ARGS[@]}"
