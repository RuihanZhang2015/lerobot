---
name: collect-so101-pick-and-place
description: Collect, resume, inspect, replay, upload, train, and run inference for Ruihan's SO-101 blue-box pick-and-place task. Use when recording LeRobot demonstrations, managing the dataset, training the ACT policy on the RTX 5090, checking training progress, or deploying the trained policy on the follower arm.
---

# Collect SO-101 Pick And Place

Run commands from `/home/ruihan/Documents/lerobot` using the `lerobot` conda environment.

The bundled scripts accept environment-variable overrides for `CONDA_BIN`,
`CONDA_ENV`, `TASK`, `REPO_ID`, `FOLLOWER_PORT`, `LEADER_PORT`, and `CAMERA`.

## Verified Setup

- Follower: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9024137-if00`, calibration ID `my_so101_follower`
- Leader: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9024308-if00`, calibration ID `my_so101_leader`
- Front camera: `/dev/video0`, 640x480 at 30 FPS
- Task: `Pick up the blue box and place it in the blue area`
- Default timing: 10-second episode, 5-second reset
- Hugging Face repo: `ruihanzhang2015/so101-pick-and-place`
- Finalized local dataset: `$HOME/.cache/huggingface/lerobot/ruihanzhang2015/so101-pick-and-place_20260612_174844`
- ACT checkpoint: `outputs/train/act_so101_pick_place/checkpoints/020000/pretrained_model`

Use the stable `/dev/serial/by-id` paths because `/dev/ttyACM0` and `/dev/ttyACM1` numbering can change. The user must belong to `dialout`; verify with `groups`. Before enabling torque, verify both ports and camera exist. Do not proceed while a motor LED is blinking or a joint/gripper is under load.

## Record

Use the bundled script:

```bash
bash ruihan_code/collect-so101-pick-and-place/scripts/record.sh
```

Optional arguments:

```bash
bash ruihan_code/collect-so101-pick-and-place/scripts/record.sh \
  --episodes 31 \
  --root "$HOME/.cache/huggingface/lerobot/ruihanzhang2015/so101-pick-and-place_TIMESTAMP" \
  --resume
```

Without `--resume`, LeRobot creates a timestamped local dataset and does not overwrite earlier recordings. With `--resume`, `--root` must identify the exact existing dataset.

Keyboard controls:

- Right Arrow: finish the current episode or reset phase early
- Left Arrow: discard and redo the current episode
- Escape: stop and finalize cleanly

Keep the launching terminal focused if global arrow keys are unreliable. Never use `Ctrl+C` to stop a valuable recording unless keyboard controls have failed; it can leave the current episode unsaved.

## Count Episodes

Read `meta/info.json` from the exact dataset root and report `total_episodes` and `total_frames`.

```bash
conda run -n lerobot python -c \
  "import json; d=json.load(open('$DATASET_ROOT/meta/info.json')); print(d['total_episodes'], d['total_frames'])"
```

## Replay

Episode indexes are zero-based. Keep the follower workspace clear.

```bash
conda run -n lerobot lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9024137-if00 \
  --robot.id=my_so101_follower \
  --dataset.repo_id=ruihanzhang2015/so101-pick-and-place \
  --dataset.root="$DATASET_ROOT" \
  --dataset.episode=1
```

## Upload

Verify authentication and metadata first, then upload the exact finalized root:

```bash
conda run -n lerobot hf auth whoami
conda run -n lerobot hf upload \
  ruihanzhang2015/so101-pick-and-place \
  "$DATASET_ROOT" . \
  --repo-type dataset \
  --commit-message "Upload SO-101 blue-box pick-and-place dataset"
```

After uploading, verify the Hub file list and confirm that metadata, parquet files, and videos are present.

## Train ACT

Use the bundled script to train on the RTX 5090 with visible terminal progress and a persistent log:

```bash
bash ruihan_code/collect-so101-pick-and-place/scripts/train_act.sh
```

Defaults:

- Batch size 64
- 20,000 steps
- Checkpoint every 5,000 steps
- Output: `outputs/train/act_so101_pick_place`
- Log: `outputs/train/act_so101_pick_place.log`

Track progress:

```bash
tail -f outputs/train/act_so101_pick_place.log
```

The script refuses to reuse an existing output directory. Set `OUTPUT_DIR` to
a new path before starting another run.

Evaluate the 20,000-step checkpoint before training longer. For this 50-episode task, 20,000–50,000 steps is a sensible range.

## Run Inference

Use the bundled script for a cautious 10-second autonomous rollout:

```bash
bash ruihan_code/collect-so101-pick-and-place/scripts/infer.sh
```

Before starting:

- Keep the follower workspace clear.
- Place the blue box in a familiar training position.
- Keep emergency power access ready.
- Use `--return_to_initial_position=false` so teardown does not introduce extra motion.

The rollout uses CUDA on the RTX 5090. ACT predicts a 100-action chunk and queues it. A single initial low-Hz warning is expected from cold inference; stop if warnings repeat or motion is choppy. Do not enable `--use_torch_compile` in this checkout: it currently fails with `Either mode or options can be specified, but both can't be specified at the same time`.
