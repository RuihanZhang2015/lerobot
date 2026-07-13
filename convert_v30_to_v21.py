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

"""Downgrade a LeRobot **v3.0** dataset to the **v2.1** layout (for openpi).

openpi pins an older lerobot (codebase_version v2.1) that cannot read v3.0
datasets, and lerobot only ships a *forward* (v2.1 -> v3.0) converter. This
script does the reverse for a *local* dataset: it re-splits the concatenated
v3.0 files back into the per-episode v2.1 layout that openpi's data loader
expects.

v3.0 (source)                              ->  v2.1 (destination)
  data/chunk-000/file-000.parquet          ->  data/chunk-000/episode_000000.parquet (one per ep)
  videos/CAM/chunk-000/file-000.mp4        ->  videos/chunk-000/CAM/episode_000000.mp4 (one per ep)
  meta/episodes/chunk-000/file-000.parquet ->  meta/episodes.jsonl
  meta/tasks.parquet                       ->  meta/tasks.jsonl
  meta/stats.json (aggregate)              ->  meta/stats.json + meta/episodes_stats.jsonl (per ep)
  meta/info.json (v3.0)                    ->  meta/info.json (v2.1)

Videos are rebuilt by decoding each episode's frames and re-encoding a fresh
per-episode MP4 (guarantees the MP4 frame count matches the parquet row count).
Temp PNGs are written per episode and deleted immediately, so peak disk use is
only one episode's worth of frames.

Run in the lerobot v3.0 environment (the one you recorded with), e.g.:

    python convert_v30_to_v21.py \
        --src ~/lerobot_data/pick_up_candy \
        --repo-id pick_up_candy

By default the v2.1 copy is written to $HF_LEROBOT_HOME/<repo_id>
(i.e. ~/.cache/huggingface/lerobot/<repo_id>), where openpi will find it by
`repo_id`. The source dataset is left untouched.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image

import av

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.video_utils import encode_video_frames

# Canonical v2.1 statistic keys. compute_episode_stats() (current lerobot) also emits
# quantile keys (q01, q10, ...); we drop those so the stats match what the old v2.1
# loader in openpi expects exactly.
V21_STAT_KEYS = {"min", "max", "mean", "std", "count"}

V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNKS_SIZE = 1000  # v2.1 default: 1000 episodes per chunk (our 31 eps all live in chunk-000)


def load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def load_episodes_meta(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def load_all_frames(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").glob("**/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("index").reset_index(drop=True)


def to_jsonable(stats: dict) -> dict:
    """Convert a {feature: {stat: np.ndarray}} dict to nested lists, keeping v2.1 keys only."""
    out = {}
    for feat, sd in stats.items():
        out[feat] = {k: np.asarray(v).tolist() for k, v in sd.items() if k in V21_STAT_KEYS}
    return out


def rebuild_videos_and_image_stats(
    src: Path,
    dst: Path,
    video_keys: list[str],
    ep_meta: pd.DataFrame,
    features: dict,
    fps: int,
) -> dict:
    """Re-split each camera's concatenated v3.0 MP4s into per-episode v2.1 MP4s.

    Returns image_stats[episode_index][video_key] = per-channel stats dict.
    """
    image_stats: dict[int, dict] = {int(e): {} for e in ep_meta["episode_index"]}

    for cam in video_keys:
        chunk_col = f"videos/{cam}/chunk_index"
        file_col = f"videos/{cam}/file_index"
        from_col = f"videos/{cam}/from_timestamp"
        # Group episodes by the source (chunk, file) they were concatenated into.
        groups = ep_meta.groupby([chunk_col, file_col], sort=True)
        for (chunk_idx, file_idx), grp in groups:
            grp = grp.sort_values(from_col)  # concatenation order == episode order
            src_video = src / "videos" / cam / f"chunk-{int(chunk_idx):03d}" / f"file-{int(file_idx):03d}.mp4"
            expected = int(grp["length"].sum())

            container = av.open(str(src_video))
            frame_iter = container.decode(video=0)

            decoded = 0
            for _, row in grp.iterrows():
                ep = int(row["episode_index"])
                length = int(row["length"])
                tmp = Path(tempfile.mkdtemp(prefix=f"v21_{cam.split('.')[-1]}_ep{ep}_"))
                png_paths = []
                for j in range(length):
                    frame = next(frame_iter)
                    img = frame.to_ndarray(format="rgb24")  # HWC uint8
                    p = tmp / f"frame-{j:06d}.png"
                    PIL.Image.fromarray(img).save(p)
                    png_paths.append(str(p))
                    decoded += 1

                out_video = dst / V21_VIDEO_PATH.format(
                    episode_chunk=ep // CHUNKS_SIZE, video_key=cam, episode_index=ep
                )
                out_video.parent.mkdir(parents=True, exist_ok=True)
                encode_video_frames(tmp, out_video, fps, overwrite=True)

                # Per-episode image stats (per-channel, in [0,1]) from the same PNGs.
                cam_stats = compute_episode_stats({cam: png_paths}, {cam: features[cam]})
                image_stats[ep][cam] = cam_stats[cam]

                shutil.rmtree(tmp, ignore_errors=True)

            container.close()
            if decoded != expected:
                raise RuntimeError(
                    f"Frame count mismatch for {src_video}: decoded {decoded}, "
                    f"expected {expected} (sum of episode lengths)."
                )
            print(f"  {cam} chunk-{int(chunk_idx):03d}/file-{int(file_idx):03d}: "
                  f"{len(grp)} episodes, {decoded} frames -> per-episode MP4s")

    return image_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to the v3.0 dataset root (has meta/ data/ videos/).")
    ap.add_argument("--repo-id", required=True, help="Repo id for the v2.1 copy (e.g. pick_up_candy).")
    ap.add_argument(
        "--dst",
        default=None,
        help="Destination root. Default: $HF_LEROBOT_HOME/<repo_id> "
        "(~/.cache/huggingface/lerobot/<repo_id>).",
    )
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    if args.dst:
        dst = Path(args.dst).expanduser().resolve()
    else:
        home = Path(__file__).home() / ".cache" / "huggingface" / "lerobot"
        dst = home / args.repo_id

    info = load_info(src)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Source is not v3.0 (got {info.get('codebase_version')}).")
    if dst.exists():
        raise SystemExit(f"Destination already exists: {dst}\nRemove it or pass a different --repo-id/--dst.")

    fps = int(info["fps"])
    features = info["features"]
    video_keys = sorted(k for k, ft in features.items() if ft["dtype"] == "video")

    ep_meta = load_episodes_meta(src)
    frames = load_all_frames(src)
    num_episodes = len(ep_meta)
    total_frames = int(ep_meta["length"].sum())
    print(f"Source: {src}\nDest:   {dst}\nEpisodes: {num_episodes} | Frames: {total_frames} | "
          f"Cameras: {video_keys}")

    (dst / "meta").mkdir(parents=True, exist_ok=True)

    # --- 1. Per-episode data parquet (non-video columns already; just split by episode) ---
    print("Writing per-episode data parquet...")
    for _, row in ep_meta.iterrows():
        ep = int(row["episode_index"])
        ep_df = frames[frames["episode_index"] == ep].reset_index(drop=True)
        out = dst / V21_DATA_PATH.format(episode_chunk=ep // CHUNKS_SIZE, episode_index=ep)
        out.parent.mkdir(parents=True, exist_ok=True)
        ep_df.to_parquet(out, index=False)

    # --- 2. Rebuild per-episode videos + collect image stats ---
    print("Rebuilding per-episode videos (decode + re-encode)...")
    image_stats = rebuild_videos_and_image_stats(src, dst, video_keys, ep_meta, features, fps)

    # --- 3. Per-episode stats (numerical + image) ---
    print("Computing per-episode stats...")
    numeric_keys = [k for k, ft in features.items() if ft["dtype"] not in ("video", "image", "string")]
    episodes_stats = []
    all_stats_for_agg = []
    for _, row in ep_meta.iterrows():
        ep = int(row["episode_index"])
        ep_df = frames[frames["episode_index"] == ep]
        num_data = {k: np.stack(ep_df[k].to_numpy()) for k in numeric_keys}
        num_features = {k: features[k] for k in numeric_keys}
        stats = compute_episode_stats(num_data, num_features)
        stats.update(image_stats[ep])  # add per-camera image stats
        all_stats_for_agg.append(stats)
        episodes_stats.append({"episode_index": ep, "stats": to_jsonable(stats)})

    # --- 4. Meta files (jsonl + aggregate stats) ---
    print("Writing meta (tasks.jsonl, episodes.jsonl, episodes_stats.jsonl, stats.json)...")
    tasks_df = pd.read_parquet(src / "meta" / "tasks.parquet").reset_index()  # index == task string
    task_col = "task" if "task" in tasks_df.columns else tasks_df.columns[0]
    with (dst / "meta" / "tasks.jsonl").open("w") as f:
        for _, r in tasks_df.sort_values("task_index").iterrows():
            f.write(json.dumps({"task_index": int(r["task_index"]), "task": r[task_col]}) + "\n")

    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for _, row in ep_meta.iterrows():
            tasks = list(row["tasks"]) if row["tasks"] is not None else []
            rec = {"episode_index": int(row["episode_index"]), "tasks": tasks, "length": int(row["length"])}
            f.write(json.dumps(rec) + "\n")

    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        for rec in episodes_stats:
            f.write(json.dumps(rec) + "\n")

    agg = aggregate_stats(all_stats_for_agg)
    (dst / "meta" / "stats.json").write_text(json.dumps(to_jsonable(agg), indent=4))

    # --- 5. info.json in v2.1 layout ---
    print("Writing v2.1 info.json...")
    v21 = dict(info)
    v21["codebase_version"] = "v2.1"
    v21.pop("data_files_size_in_mb", None)
    v21.pop("video_files_size_in_mb", None)
    v21["data_path"] = V21_DATA_PATH
    v21["video_path"] = V21_VIDEO_PATH
    v21["chunks_size"] = CHUNKS_SIZE
    v21["total_episodes"] = num_episodes
    v21["total_frames"] = total_frames
    v21["total_tasks"] = int(tasks_df["task_index"].nunique())
    v21["total_videos"] = num_episodes * len(video_keys)
    v21["total_chunks"] = (num_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE
    v21["splits"] = {"train": f"0:{num_episodes}"}
    # v2.1 non-video features do not carry a per-feature "fps" field.
    for k, ft in v21["features"].items():
        if ft.get("dtype") != "video":
            ft.pop("fps", None)
    (dst / "meta" / "info.json").write_text(json.dumps(v21, indent=4))

    print(f"\nDone. v2.1 dataset written to:\n  {dst}\n"
          f"Use repo_id='{args.repo_id}' in your openpi config "
          f"(ensure HF_LEROBOT_HOME points at {dst.parent}).")


if __name__ == "__main__":
    main()
