#!/usr/bin/env python
"""Dual-arm ACT process_data (UniVTAC dual_* tasks).

Reads slim (or raw) dual hdf5 and emits ACT-format per-episode hdf5, storing
images as ENCODED JPEG bytes (byte-copied from source, no decode/re-encode) so
the on-disk footprint stays ~slim size. The ACT dataloader (patched
TacArenaDataset) decodes on load.

Action/qpos semantics mirror envs/utils/data.py batch_gather_hdf5:
  state[t]  = joint[t]      (data[:-1][ds_idx])
  action[t] = joint[t+1]    (data[1:][ds_idx])
Per arm we take joint[:, 0:8] (7 arm joints + gripper finger0). Dual => 16 dim.

Usage:
  process_data_dual.py <task_name> <task_config> <expert_data_num> <src_dir> [downsample]
    <src_dir> contains hdf5/*.hdf5  (e.g. ~/yifan_ws/slim_raw/dual_bowl_unstack)
Outputs to  $ACT_DATA_ROOT/sim-<task>/<config>-<num>/episode_i.hdf5
or ~/yifan_ws/act_data/sim-<task>/<config>-<num>/episode_i.hdf5 by default
and registers SIM_TASK_CONFIGS.json in the current dir (run from policy/ACT).
"""
import sys, os, h5py, numpy as np, argparse, json
from pathlib import Path

CAM_SRC = {
    "cam_high": "observation/head/rgb",
    "cam_wrist": "observation/wrist/rgb",
    "cam_wrist_b": "observation/wrist_b/rgb",
}
TAC_SRC = {
    "tac_left": "tactile/left_tactile/rgb_marker",
    "tac_right": "tactile/right_tactile/rgb_marker",
    "tac_left_b": "tactile/left_tactile_b/rgb_marker",
    "tac_right_b": "tactile/right_tactile_b/rgb_marker",
}
JOINT_DIM = 8  # per arm: 7 joints + gripper finger0 (matches single-arm [0:8])
OUT_ROOT = os.path.expanduser(os.environ.get("ACT_DATA_ROOT", "~/yifan_ws/act_data"))


def process_one(src, dst, ds):
    with h5py.File(src, "r") as f:
        jA = f["embodiment/joint"][()]      # (T,9)
        jB = f["embodiment_b/joint"][()]    # (T,9)
        T = jA.shape[0]
        idx = np.arange(0, T - 1, ds)       # obs/state indices; action = next joint
        qpos = np.concatenate(
            [jA[:-1][idx][:, :JOINT_DIM], jB[:-1][idx][:, :JOINT_DIM]], axis=1
        ).astype(np.float32)                # (n,16)
        action = np.concatenate(
            [jA[1:][idx][:, :JOINT_DIM], jB[1:][idx][:, :JOINT_DIM]], axis=1
        ).astype(np.float32)                # (n,16)
        imgs = {}
        for k, srckey in {**CAM_SRC, **TAC_SRC}.items():
            arr = f[srckey][()]             # (T,) S-dtype encoded JPEG bytes
            imgs[k] = arr[:-1][idx]         # (n,) aligned to obs/state
    with h5py.File(dst, "w") as fo:
        fo.create_dataset("action", data=action)
        obs = fo.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        im = obs.create_group("images")
        for k, arr in imgs.items():
            im.create_dataset(k, data=arr)  # preserves S-dtype (encoded bytes)
    return qpos.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_name")
    ap.add_argument("task_config")
    ap.add_argument("expert_data_num", type=int)
    ap.add_argument("src_dir", help="dir containing hdf5/*.hdf5")
    ap.add_argument("downsample", nargs="?", type=int, default=1)
    a = ap.parse_args()

    src_hdf5 = Path(a.src_dir) / "hdf5"
    if not src_hdf5.exists():
        src_hdf5 = Path(a.src_dir)
    files = sorted(src_hdf5.glob("*.hdf5"), key=lambda p: int(p.stem))
    assert a.expert_data_num <= len(files), \
        f"requested {a.expert_data_num} > found {len(files)}"

    out_dir = Path(OUT_ROOT) / f"sim-{a.task_name}" / f"{a.task_config}-{a.expert_data_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dual-process] {a.task_name} src={src_hdf5} n={a.expert_data_num} ds={a.downsample}", flush=True)
    lens = []
    for i in range(a.expert_data_num):
        n = process_one(str(files[i]), str(out_dir / f"episode_{i}.hdf5"), a.downsample)
        lens.append(n)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{a.expert_data_num}] last_len={n}", flush=True)
    print(f"[done] {a.expert_data_num} eps, frames min/max/mean="
          f"{min(lens)}/{max(lens)}/{int(sum(lens)/len(lens))} -> {out_dir}", flush=True)

    key = f"sim-{a.task_name}-{a.task_config}-{a.expert_data_num}"
    cfg_path = "./SIM_TASK_CONFIGS.json"
    try:
        cfg = json.load(open(cfg_path))
    except Exception:
        cfg = {}
    cfg[key] = {
        "dataset_dir": str(out_dir),
        "num_episodes": a.expert_data_num,
        "episode_len": 1000,
        "camera_names": list(CAM_SRC),
        "tactile_names": list(TAC_SRC),
    }
    json.dump(cfg, open(cfg_path, "w"), indent=4)
    print(f"[registered] {key} -> {cfg_path}", flush=True)


if __name__ == "__main__":
    main()
