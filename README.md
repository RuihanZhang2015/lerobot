# SO-101 candy picking — pi0.5 LoRA experiments

A working fork of [LeRobot](https://github.com/huggingface/lerobot) used to teach an SO-101 arm
to pick up a candy and drop it in a container, with a [pi0.5](https://github.com/Physical-Intelligence/openpi)
policy fine-tuned via LoRA.

The task string is the same everywhere and must stay that way — the policy is prompted with it at
inference:

> `Pick up the object and place it in the container`

Upstream LeRobot docs live at [huggingface.co/docs/lerobot](https://huggingface.co/docs/lerobot);
this README covers only what is specific to these experiments. For general SO-101 setup
(calibration, ports, cameras) see [`AGENT_GUIDE.md`](./AGENT_GUIDE.md).

---

## Hardware

| Part | Value |
| --- | --- |
| Follower | SO-101, `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9024137-if00` |
| Leader | SO-101, `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9024308-if00` |
| Cameras | `top` = `/dev/video0`, `wrist` = `/dev/video2` — both 1280×720 MJPG @ 30fps |
| GPU | RTX 5090, 32 GB |

Use the `by-id` paths, not `/dev/ttyACM*` — the ACM numbering swaps between reboots and will
happily point the follower driver at the leader arm.

Two Python environments are in play, and mixing them up is the single most common failure:

| Env | Used for | Interpreter |
| --- | --- | --- |
| `lerobot-src` | recording, conversion, eval client | `~/miniforge3/envs/lerobot-src/bin/python` |
| openpi `.venv` | policy server, training | `cd ~/ke/openpi && uv run ...` |

A plain `python` from the wrong conda env fails with
`ImportError: 'datasets' is required but not installed`.

---

## The scripts in this repo

| Script | What it does |
| --- | --- |
| [`record_ab.py`](./record_ab.py) | Press-A-to-start / press-B-to-stop recording. The arm is frozen until you press A, so the scene can be reset between demos. |
| [`record_recovery.py`](./record_recovery.py) | Adds a **staging** phase: `C` drives the arm via teleop *without recording*, so you can put the robot and scene into a failed state, then `A` records only the recovery from it. |
| [`convert_v30_to_v21.py`](./convert_v30_to_v21.py) | Downgrades a LeRobot v3.0 dataset to the v2.1 layout. openpi pins an older lerobot that cannot read v3.0, and lerobot only ships the forward converter. |
| [`eval_so101_openpi.py`](./eval_so101_openpi.py) | Real-time eval bridge: streams observations to an openpi policy server over websocket and executes the returned action chunks on the arm. |

### Why the recovery demos exist

A policy trained only on clean picks has never seen a dropped candy or a bad grasp pose, so when it
fails it has no idea what to do next. `record_recovery.py` exists to fix that: you hand-stage the
failure (knock the candy over, drag the arm somewhere useless) with teleop while nothing is being
written, and then record *only* the recovery. Staging calls `record_loop` with `dataset=None`, so no
frames land in the dataset.

Cycle: `[idle] --C--> [teleop, not recording] --A--> [recording] --B--> save`

`C` is refused when no teleoperator is configured — `record_loop`'s no-teleop branch skips its sleep
and would busy-spin.

---

## Datasets collected

All are v3.0, 30fps, 6-DoF action + state, two 720p cameras, `so_follower`, one shared task string.

| Dataset | Episodes | Frames | Notes |
| --- | --- | --- | --- |
| `pick_up_candy` | 31 | 13,370 | First batch. This is what the current checkpoint was trained on. |
| `pick_candy2` | 13 | 5,759 | |
| `pick_candy3` | 9 | 3,593 | |
| `pick_candy_recovery` | 6 | 3,301 | Failure-recovery demos, staged with `record_recovery.py`. |
| **`pick_candy_all`** | **59** | **26,023** | Merge of all four, for the next training run. |

They live under `~/lerobot_data/<name>` (v3.0); the v2.1 copies openpi trains from land in
`~/.cache/huggingface/lerobot/<repo_id>`.

### Recording

```bash
~/miniforge3/envs/lerobot-src/bin/python record_recovery.py \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9024137-if00 \
  --robot.id=so101_follower_main \
  --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video2", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9024308-if00 \
  --teleop.id=so101_leader_main \
  --dataset.repo_id=local/pick_candy_recovery2 \
  --dataset.root="$HOME/lerobot_data/pick_candy_recovery2" \
  --dataset.num_episodes=20 --dataset.episode_time_s=3600 --dataset.fps=30 \
  --dataset.single_task="Pick up the object and place it in the container" \
  --dataset.push_to_hub=false --display_data=true
```

Keep the leader and follower roughly aligned before pressing `C` or `A` — the follower snaps to the
leader's pose on the first frame of each phase. `--robot.max_relative_target` clamps how far a single
step may move if that worries you.

Pasting a multi-line command with trailing backslashes into some terminals mangles the later flags
(bracketed paste). If flags come out garbled, paste it as one line.

---

## Merging + converting

`merge_datasets` (from `lerobot.datasets.dataset_tools`) concatenates the v3.0 datasets and reindexes
episodes; the four sources are schema-identical, so they merge without fixups. Then the merged set is
downgraded to v2.1 for openpi:

```bash
~/miniforge3/envs/lerobot-src/bin/python convert_v30_to_v21.py \
  --src ~/lerobot_data/pick_candy_all --repo-id pick_candy_all
```

The conversion decodes and re-encodes every episode's video for both cameras (guaranteeing the MP4
frame count matches the parquet row count), so it takes a while — it is the slow step, not the
training setup.

---

## Training

openpi config `pi05_so101_lora_all` in `~/ke/openpi/src/openpi/training/config.py`: pi0.5 with LoRA on
both the VLM (`gemma_2b_lora`) and the action expert (`gemma_300m_lora`), delta actions, action horizon
50, batch size 32, 10k steps with a cosine schedule (1k warmup, peak 5e-5, decaying to 5e-6).

It is a **separate config** from the original `pi05_so101_lora` (which is pinned to `pick_up_candy`, 31
episodes), so the existing checkpoint stays reproducible.

```bash
cd ~/ke/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_so101_lora_all
uv run scripts/train.py pi05_so101_lora_all --exp-name=pick_candy_all_lora
```

Free the GPU first — a running policy server holds ~30 GB of the 32 GB card, and training will OOM
behind it.

If the cosine `decay_steps` is left at 20k while `num_train_steps` is 10k, the LR gets truncated
mid-decay instead of landing at `decay_lr`. Change them together.

---

## Eval

Serve the checkpoint (openpi env):

```bash
cd ~/ke/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101_lora \
  --policy.dir=~/ke/openpi/checkpoints/pi05_so101_lora/pick_up_candy_lora/6000
```

Then drive the arm (lerobot-src env):

```bash
~/miniforge3/envs/lerobot-src/bin/python -u eval_so101_openpi.py \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9024137-if00 \
  --robot.id=so101_follower_main \
  --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video2", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}}' \
  --task="Pick up the object and place it in the container" \
  --host=localhost --port=8000 --fps=30 --duration_s=30
```

The server owns image resize, normalization, delta→absolute conversion, and slicing back to the 6
SO-101 joints, so the client only ships raw observations and executes what comes back.

Measured on the step-6000 checkpoint: chunks of shape `(50, 6)` returned in **77–81 ms**, and the loop
held 30fps for the full 900 steps of a 30s rollout. `--actions_per_chunk` (default 25 of the 50-step
horizon) trades responsiveness against stability.

### If eval prints nothing, it is still driving the arm

`logging.basicConfig()` is a no-op when a root handler already exists — which it does by the time
`eval_main` runs, thanks to the rerun/draccus/lerobot imports — so the root logger stays at WARNING and
every progress line vanishes. A silent run looks exactly like a crashed no-op run while the robot is in
fact moving through the full episode. Fixed with `force=True` in `eval_so101_openpi.py`; if you ever
see a silent-but-slow run again, treat it as **live, not dead** — check the arm before re-running.

---

## Status

- Current checkpoint: `checkpoints/pi05_so101_lora/pick_up_candy_lora/6000`, trained on `pick_up_candy`
  only (31 episodes). It serves and runs the arm at 30fps; real-world success rate has not been
  measured systematically.
- `pick_candy_all` (59 episodes) is merged and converting to v2.1. Norm stats and the 10k-step training
  run have not been done yet.
- `pick_candy_recovery2` was set up but never recorded — the dataset does not exist.
- The openpi-side changes (`pi05_so101_lora_all` config, `so101_policy.py`) are uncommitted: that
  checkout's `origin` is `Physical-Intelligence/openpi` upstream, with no personal fork configured.
