#!/usr/bin/env bash

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
CONDA_ENV="${CONDA_ENV:-lerobot}"

TASK="${TASK:-Pick up the blue box and place it in the blue area}"
REPO_ID="${REPO_ID:-ruihanzhang2015/so101-pick-and-place}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9024137-if00}"
LEADER_PORT="${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9024308-if00}"
CAMERA="${CAMERA:-/dev/video0}"

if [[ -z "$CONDA_BIN" ]]; then
  echo "conda was not found. Set CONDA_BIN to its executable path." >&2
  exit 1
fi

require_paths() {
  local path
  for path in "$@"; do
    if [[ ! -e "$path" ]]; then
      echo "Required path is missing: $path" >&2
      exit 1
    fi
  done
}
