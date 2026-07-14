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

"""Recorder for failure-recovery demos: stage a failure with teleop, then record the recovery.

A variant of ``record_ab.py``. The difference is a **staging** phase (C) in which the
leader arm drives the follower but *nothing is written to the dataset*. Use it to put the
robot and the scene into the failed state you want to recover from — knock the object over,
drag the arm into a bad pose — then press A and the episode records only the recovery.

Each cycle:

    [idle: arm frozen]
        C   -> teleop live, NOT recording   (stage the failure)
        A   -> start recording the episode  (also works straight from idle, for normal demos)
        B   -> stop the demo (it is saved) and go back to idle
        R   -> discard the current demo and re-record it
        Esc/Q -> stop recording, save everything, and quit

Teleop is off unless you ask for it: the follower is only driven during C (staging) and
during a recording episode. Note that the follower snaps to the leader's pose on the first
frame after C or A, so keep the two arms roughly aligned — or set
``--robot.max_relative_target`` to clamp how far a single step may move.

Same CLI flags as ``lerobot-record`` (``--robot.*``, ``--teleop.*``, ``--dataset.*``,
``--display_data`` …). ``--dataset.episode_time_s`` still caps a demo if you forget to press
B. Staging has no time cap.

Run it with the project's interpreter, e.g.:

    python record_recovery.py \
        --robot.type=so101_follower --robot.port=/dev/ttyACM1 \
        --robot.id=so101_follower_main \
        --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video2", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}}' \
        --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 \
        --teleop.id=so101_leader_main \
        --dataset.repo_id=local/so101_pick_candy_recovery \
        --dataset.root="$HOME/lerobot_data/pick_candy_recovery" \
        --dataset.num_episodes=50 --dataset.episode_time_s=3600 \
        --dataset.single_task="Pick up the object and place it in the container" \
        --dataset.push_to_hub=false --display_data=true
"""

import logging
import time
from dataclasses import asdict
from pprint import pformat

from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config

# Importing the record module registers every robot/teleop/camera choice with draccus
# (its module-level imports have that side effect) and gives us RecordConfig + record_loop.
from lerobot.scripts.lerobot_record import RecordConfig, record_loop
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import (
    init_visualization,
    shutdown_visualization,
)

# How long a single staging `record_loop` call runs before we re-enter it. Staging is
# meant to be open-ended, so we just loop; this only bounds how long a chunk lasts.
STAGING_CHUNK_S = 60


def _make_listener(has_teleop: bool):
    """Start a keyboard listener wired to the C/A/B/R/Esc control flow.

    Returns ``(listener, events, state)``. ``events`` uses the keys ``record_loop`` expects
    (it only reads/clears ``exit_early``) plus our own flags. ``state["mode"]`` is one of
    ``idle`` / ``staging`` / ``recording`` and makes the dispatch mode-aware, so that e.g. B
    can't kill an episode that isn't running and A can't cut one short.
    """
    events = {
        "start_episode": False,  # A — begin the episode
        "exit_early": False,  # read + cleared by record_loop to break its inner loop
        "rerecord_episode": False,  # R — discard + redo the current demo
        "stop_recording": False,  # Esc/Q — quit and save
        "staging": False,  # C — teleop live, nothing recorded
    }
    state = {"mode": "idle"}

    def dispatch(name: str) -> None:
        key = name.lower()
        mode = state["mode"]

        if key == "c":
            if not has_teleop:
                # record_loop's no-teleop branch busy-spins; never enter staging without a leader.
                print("C ignored — no --teleop configured, so there is nothing to drive the arm.")
                return
            if mode == "idle":
                print("C pressed — teleop live (not recording). Stage the failure, then press A.")
                events["staging"] = True

        elif key == "a":
            if mode in ("idle", "staging"):
                events["start_episode"] = True
                events["exit_early"] = True  # break the staging record_loop, if we're in one

        elif key in ("b", "right", "n"):
            if mode == "recording":
                print("B pressed — stopping current demo.")
                events["exit_early"] = True

        elif key in ("r", "left"):
            if mode == "recording":
                print("R pressed — re-recording current demo.")
                events["rerecord_episode"] = True
                events["exit_early"] = True

        elif key in ("esc", "q"):
            print("Esc pressed — stopping data recording.")
            events["stop_recording"] = True
            events["exit_early"] = True

    listener = create_key_listener(
        dispatch, controls_help="C=teleop/stage, A=start, B=stop, R=re-record, Esc=quit"
    )
    return listener, events, state


@parser.wrap()
def record_recovery(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_visualization(
            cfg.display_mode, session_name="recording", ip=cfg.display_ip, port=cfg.display_port
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    if teleop is None:
        logging.warning(
            "No --teleop configured: the C staging phase is unavailable and episodes will record "
            "no actions. Pass --teleop.type=... to drive the arm."
        )

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    # Everything record_loop needs that doesn't change between phases. The only difference
    # between staging and recording is `dataset` (None => nothing is written) and the cap.
    loop_kwargs = dict(
        robot=robot,
        fps=cfg.dataset.fps,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        single_task=cfg.dataset.single_task,
        display_data=cfg.display_data,
        display_mode=cfg.display_mode,
        display_compressed_images=display_compressed_images,
    )

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events, state = _make_listener(has_teleop=teleop is not None)

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                # ---- Idle: arm frozen, waiting for C (stage) or A (record) ----
                state["mode"] = "idle"
                # Clear flags left over from the previous demo so a stray B/R can't kill the
                # next episode before its first frame is captured.
                events["start_episode"] = False
                events["exit_early"] = False
                events["rerecord_episode"] = False
                events["staging"] = False
                log_say(
                    "Press C to teleop without recording, A to start recording, Escape to quit.",
                    cfg.play_sounds,
                )
                while not (events["start_episode"] or events["staging"] or events["stop_recording"]):
                    time.sleep(0.02)
                if events["stop_recording"]:
                    break

                # ---- Staging (C): teleop drives the arm, dataset=None so nothing is written.
                # Ends only when A starts the episode or Esc quits. Each call is capped at
                # STAGING_CHUNK_S, so we re-enter until one of those flags flips.
                while events["staging"] and not (events["start_episode"] or events["stop_recording"]):
                    state["mode"] = "staging"
                    events["exit_early"] = False
                    record_loop(
                        events=events,
                        dataset=None,
                        control_time_s=STAGING_CHUNK_S,
                        **loop_kwargs,
                    )
                if events["stop_recording"]:
                    break

                # ---- Record until B / R / Esc / the episode_time_s safety cap ----
                state["mode"] = "recording"
                events["start_episode"] = False
                events["exit_early"] = False
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    events=events,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    **loop_kwargs,
                )
                state["mode"] = "idle"

                # ---- Re-record (R): drop this demo and loop back to idle ----
                if events["rerecord_episode"]:
                    log_say("Re-record: discarding episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                # ---- Guard against an empty demo (B/Esc pressed before any frame) ----
                if not dataset.has_pending_frames():
                    log_say("Empty episode, discarding", cfg.play_sounds)
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                # dataset.num_episodes is now the new global total; the episode just saved got
                # index (total - 1). Report both so resumed sessions aren't confusing.
                log_say(
                    f"Saved episode {dataset.num_episodes - 1} "
                    f"(dataset total: {dataset.num_episodes}, this session: {recorded_episodes})",
                    cfg.play_sounds,
                )
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if listener is not None:
            listener.stop()

        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record_recovery()


if __name__ == "__main__":
    main()
