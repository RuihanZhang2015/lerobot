#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-time eval bridge: drive an SO-101 follower with an openpi pi0.5 policy server.

Run the policy server in the openpi environment first:

    cd ~/ke/openpi
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_so101_lora \
        --policy.dir=/home/ruihan/ke/openpi/checkpoints/pi05_so101_lora/pick_up_candy_lora/6000

Then run this script in the lerobot (recording) environment, with the same robot args
you used to record (port / id / cameras):

    conda activate lerobot-src
    pip install -e ~/ke/openpi/packages/openpi-client   # one-time
    python eval_so101_openpi.py \
        --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=so101_follower_main \
        --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video2", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}}' \
        --task="Pick up the object and place it in the container"

The server handles image resize, normalization, delta->absolute action conversion, and
slicing back to the 6 SO-101 joints, so this client only sends raw observations
(``observation/image`` = top cam, ``observation/wrist_image`` = wrist cam,
``observation/state`` = 6 joint positions, ``prompt``) and executes the returned
``actions`` (shape ``[action_horizon, 6]``).

Press Ctrl-C to stop safely (torque is released on disconnect).
"""

import logging
import threading
import time
from dataclasses import dataclass

import draccus
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  (registers "opencv")
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401  (registers "realsense")
from lerobot.robots import (  # noqa: F401  (importing registers the robot choices)
    RobotConfig,
    make_robot_from_config,
    so_follower,
)
from openpi_client import websocket_client_policy


@dataclass
class EvalConfig:
    # SO-101 follower robot (same --robot.* args you recorded with).
    robot: RobotConfig
    # Language instruction sent to the policy (should match your dataset task).
    task: str = "Pick up the object and place it in the container"
    # Policy server address.
    host: str = "localhost"
    port: int = 8000
    # Control rate (must match the dataset fps the policy was trained on).
    fps: int = 30
    # Receding horizon: execute this many actions from each predicted chunk, then re-query.
    # The model predicts action_horizon (=50) steps; 25 gives a good responsiveness/stability trade-off.
    actions_per_chunk: int = 25
    # How long to run before stopping (seconds).
    duration_s: float = 60.0
    # Camera keys (as named in --robot.cameras) mapped to the model's image slots.
    base_camera: str = "top"
    wrist_camera: str = "wrist"
    # Show live camera feeds in a rerun viewer (OpenCV in this env is headless, so we use
    # rerun instead of cv2 windows). Both cameras appear as separate live panels.
    show_cameras: bool = True
    # Refresh rate of the live camera panels (independent of the control loop).
    display_fps: int = 20


def _camera_display_loop(robot, cam_keys, stop_event, fps):
    """Background thread: stream the cameras' latest frames into the rerun viewer.

    Reads ``camera.read_latest()`` — a thread-safe snapshot of the frame the camera's
    own capture thread already grabbed — so this never competes with the control loop
    for the video device. Frames are RGB HWC uint8, which rerun expects directly.
    """
    import rerun as rr

    period = 1.0 / fps
    while not stop_event.is_set():
        t0 = time.perf_counter()
        for cam in cam_keys:
            try:
                frame = robot.cameras[cam].read_latest(max_age_ms=1000)
                rr.log(f"cameras/{cam}", rr.Image(frame))
            except Exception:  # a momentarily stale/missing frame shouldn't kill the viewer
                pass
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))


@draccus.wrap()
def eval_main(cfg: EvalConfig):
    # force=True: an import (rerun/draccus/lerobot) may already have attached a root handler,
    # which would make basicConfig a silent no-op and leave the root logger at WARNING —
    # dropping every progress line below.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # Start the live camera viewer (rerun) on a background thread.
    stop_display = threading.Event()
    display_thread = None
    if cfg.show_cameras:
        import rerun as rr

        rr.init("so101_eval", spawn=True)
        display_thread = threading.Thread(
            target=_camera_display_loop,
            args=(robot, [cfg.base_camera, cfg.wrist_camera], stop_display, cfg.display_fps),
            daemon=True,
        )
        display_thread.start()
        logging.info("Live camera viewer (rerun) started: %s + %s", cfg.base_camera, cfg.wrist_camera)

    # State/action order = the robot's ".pos" observation keys, in the robot's motor order.
    # This is the same order the dataset was recorded with, so it matches the trained policy.
    state_keys = [k for k in robot.observation_features if isinstance(k, str) and k.endswith(".pos")]
    logging.info("Joint order (%d): %s", len(state_keys), state_keys)

    for cam in (cfg.base_camera, cfg.wrist_camera):
        if cam not in robot.cameras:
            raise ValueError(
                f"Camera '{cam}' not found in robot cameras {list(robot.cameras)}. "
                f"Set --base_camera / --wrist_camera to match your --robot.cameras keys."
            )

    client = websocket_client_policy.WebsocketClientPolicy(host=cfg.host, port=cfg.port)
    logging.info("Connected to policy server at %s:%d. Task: %r", cfg.host, cfg.port, cfg.task)

    period = 1.0 / cfg.fps
    total_steps = int(cfg.duration_s * cfg.fps)
    executed = 0
    try:
        while executed < total_steps:
            # Query the policy with the current observation.
            obs = robot.get_observation()
            state = np.array([obs[k] for k in state_keys], dtype=np.float32)
            request = {
                "observation/image": obs[cfg.base_camera],
                "observation/wrist_image": obs[cfg.wrist_camera],
                "observation/state": state,
                "prompt": cfg.task,
            }
            t_infer = time.perf_counter()
            action_chunk = np.asarray(client.infer(request)["actions"])  # (action_horizon, 6)
            logging.info(
                "Inferred chunk %s in %.0f ms (executed %d/%d)",
                tuple(action_chunk.shape),
                (time.perf_counter() - t_infer) * 1e3,
                executed,
                total_steps,
            )

            # Execute part of the chunk open-loop, then re-query (receding horizon).
            n = min(cfg.actions_per_chunk, len(action_chunk))
            for i in range(n):
                t0 = time.perf_counter()
                row = action_chunk[i]
                action = {k: float(row[j]) for j, k in enumerate(state_keys)}
                robot.send_action(action)
                executed += 1
                if executed >= total_steps:
                    break
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        stop_display.set()
        if display_thread is not None:
            display_thread.join(timeout=1.0)
        robot.disconnect()
        logging.info("Stopped. Executed %d steps.", executed)


def main():
    eval_main()


if __name__ == "__main__":
    main()
