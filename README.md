<h1 align="center">UniVTAC</h1>

> UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking<br>
> [arXiv](https://arxiv.org/abs/2602.10093) | [PDF](https://arxiv.org/pdf/2602.10093) | [Website](https://univtac.github.io/) | [HuggingFace Dataset](https://huggingface.co/datasets/byml/UniVTAC) | [Modelscope Dataset](https://modelscope.cn/datasets/byml2024/UniVTAC)

**UniVTAC** is a tactile-aware simulation benchmark for robotic manipulation built on top of **NVIDIA Isaac Lab** and **TacEx (UIPC-based tactile simulation)**. It provides a unified framework for collecting expert demonstrations, training visuotactile policies, and evaluating them across a diverse suite of contact-rich manipulation tasks — all with high-fidelity tactile feedback from simulated GelSight Mini, ViTai GF225, or XenseWS sensors.

## Installation

See the [Installation Guide](./docs/Installation.md) for detailed setup instructions, including installing the environment, installing TacEx from the modified local source and setting up cuRobo for motion planning.

## Task Gallery

UniVTAC currently includes the following manipulation tasks, all featuring tactile sensing:

| Task | Module | Description |
|---|---|---|
| **Collect** | `collect` | Collect contact-rich tactile data for pretraining |
| **Lift Bottle** | `lift_bottle` | Grasp and lift a bottle off a surface near a wall |
| **Lift Can** | `lift_can` | Grasp and lift a cylindrical can |
| **Insert HDMI** | `insert_HDMI` | Insert an HDMI connector into a port |
| **Insert Hole** | `insert_hole` | Precision peg-in-hole insertion |
| **Insert Tube** | `insert_tube` | Insert a tube into a fixture |
| **Insert USB** | `insert_USB` | Peg-in-hole insertion of a USB connector into a slot |
| **Pull Out Key** | `pull_out_key` | Extract a key from a lock |
| **Put Bottle in Shelf** | `put_bottle_in_shelf` | Place a bottle onto a shelf |
| **Grasp & Classify** | `grasp_classify` | Grasp an object and classify it by tactile feedback |
| **Grasp Chip** | `grasp_chip` | Grasp a fragile chip without crushing it (tactile force limit) |
| **Pour Ball** | `pour_ball` | Grasp a cup and pour the balls inside it out onto a plate |
| **Phone Socket Replug** | `phone_socket_replug` | Pull a phone connector out of its socket and plug it back in |
| **Dual-Arm Screw & Sleeve** | `dual_screw_sleeve` | Dual-arm assembly: one arm holds the sleeve, the other inserts a screw into it |
| **Dual-Arm Cup Stack** | `dual_cup_stack` | Dual-arm nesting: one arm holds the bottom cup, the other stacks a second cup into it |
| **Dual-Arm Bowl Unstack** | `dual_bowl_unstack` | Dual-arm: grasp the top bowl of a nested pair by its rim, lift it out, and set it aside |
| **Dual-Arm Gear Holder** | `dual_gear_holder` | Dual-arm: pick gears laid around a holder and thread each onto one of the holder's vertical pegs |
| **Dual-Arm Bowl Place-Stack** | `dual_bowl_place_stack` | Dual-arm: bring two separated bowls together into a centered nested stack |
| **Dual-Arm Cup Place-Stack** | `dual_cup_place_stack` | Dual-arm: bring two separated cups together into a centered nested stack |
| **Dual-Arm Cup Handover & Place** | `dual_cup_handover_place` | Dual-arm: one arm grasps a cup and hands it to the other arm, which places it |
| **Dual-Arm Plate Place-Stack** | `dual_plate_place_stack` | Dual-arm: bring two separated plates together into a centered stack |

To build more tasks, refer to the [Task Creation Guide](./docs/TaskCreation.md) for instructions on how to define new manipulation tasks within the UniVTAC framework — including dual-arm tasks, rigid in-hand "weld" carrying, rim grasps for wide objects, and peg insertion.

## Data Collection

See the [Data Collection Guide](./docs/Collection.md) for instructions on how to run the automated data collection pipeline, configure task-specific parameters, and understand the output data structure.

Dataset containing 100 episodes per task can be downloaded from [HuggingFace](https://huggingface.co/datasets/byml/UniVTAC), [Modelscope](https://modelscope.cn/datasets/byml2024/UniVTAC) or by running the script in `data/download.sh`.

## Visualizing Tasks & Saving Videos

Every task runs through the same entry point, `scripts/collect_data.py`, driven by a YAML config in `task_config/`. The config controls whether a video is recorded and whether an interactive Isaac Sim window opens.

```bash
python scripts/collect_data.py <task_name> <task_config> [--gpu <id>]
# e.g.
python scripts/collect_data.py insert_USB demo
```

- `<task_name>` — a module under `envs/` (e.g. `insert_USB`, `lift_can`, `dual_screw_sleeve`).
- `<task_config>` — a YAML under `task_config/` (extension optional, so `demo` resolves to `task_config/demo.yml`).

### Saving a task's video

Run the task with a config whose `video_frequency > 0`. The MP4 is written to:

```
data/<task_name>/<task_config>/video/<seed>_<result>.mp4
```

where `<result>` is `success`, `fail`, or `error` — so the clip is saved **no matter the outcome**. Each frame tiles the head + wrist camera views alongside the tactile-pad images.

Config keys that control recording (see `task_config/demo.yml` for a full example):

| Key | Meaning |
|---|---|
| `video_frequency` | `>0` enables MP4 recording (writes one frame every N steps); `0` disables it |
| `render_frequency` | `0` = headless (no window, pure recording); `1` = also open a live Isaac Sim window |
| `save_frequency` | cadence for saving HDF5 observations |
| `episode_num` | number of **successful** episodes to collect before stopping |
| `start_seed` / `max_seed` | first seed / seed cap (CLI `--start_seed` / `--max_seed` override the YAML) |
| `observations` | which `camera` / `tactile` / `embodiment` / `actor` signals to log |

To record a single **try shot** — run one seed exactly once at the highest frequency (a frame every step) and save the video whether it succeeds or fails — use the ready-made `task_config/record_one.yml` (`episode_num: 1`, `start_seed: 0`, `max_seed: 0`, `video_frequency: 1`):

```bash
python scripts/collect_data.py <task_name> record_one
# -> data/<task_name>/record_one/video/0_<result>.mp4
```

### Poster / figure view (third-person + tactile only)

By default each video frame tiles the head **and wrist** camera views next to the tactile pads. For figures or posters you often want only the third-person (head) view plus the tactile views, with the wrist cameras removed. Set `UNIVTAC_POSTER_VIEW=1` to drop the wrist panels:

```bash
UNIVTAC_POSTER_VIEW=1 python scripts/collect_data.py <task_name> record_one
```

The frame then contains **head + tactile** only and is sized to fit: single-arm tasks → `640x320` (head + 2 tactiles), dual-arm tasks → `800x320` (head + 4 tactiles). Everything else (which seeds run, where the MP4 is written) is unchanged.

### Visualizing in the Isaac Sim GUI

All three options below open the live Isaac Sim window and therefore require a display (`DISPLAY` set):

1. **Watch while collecting** — set `render_frequency: 1` in the config (see `task_config/gui.yml`). This opens an Isaac Sim window with live rendering as the scripted demo runs, and still records the video:
   ```bash
   python scripts/collect_data.py insert_USB gui
   ```
2. **Inspect a task scene** (robot + objects, no scripted motion) — handy while building or placing a new task:
   ```bash
   python scripts/view_task.py <task_name> [--seed <id>]   # e.g. python scripts/view_task.py insert_USB
   ```
3. **Inspect a single asset** (`.usd`) on a ground plane:
   ```bash
   python scripts/view_usd.py assets/objects/USB.usd
   ```

> **Multi-GPU / headless note:** Isaac Sim's renderer must run on an RTX NVIDIA GPU. On machines whose X display defaults to a non-NVIDIA GPU (e.g. an Intel iGPU), force the NVIDIA Vulkan driver before launching — this is also required for headless camera/video recording:
> ```bash
> VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
>   __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia DISPLAY=:1 \
>   python scripts/collect_data.py insert_USB gui
> ```

## Train & Eval Policies

UniVTAC includes several baseline policies implemented under the `policy/` directory:

- ACT: Action Chunking with Transformers with/without tactile inputs
- Abation: ACT ablation variants for modality comparison
- ViTAL: ACT with CLIP-pretrained tactile-vision encoders in ViTAL

Each policy is a self-contained module under `policy/` with its own data processing, training, and deployment scripts. All policies share a unified evaluation entry point at the project root:

```bash
bash eval_policy.sh ${task_name} ${task_config} ${policy_config} ${gpu_id}
```

For parallel evaluation over many seeds:

```bash
bash parallel_eval.sh ${task_name} ${task_config} ${policy_config} ${gpu_id} [num_processes] [total_num]
```

The evaluation results, including videos and success rate logs, will be saved in the `eval_result/` directory under the project root.

To deploy your own policy, refer to the [Deploy Your Policy](./docs/Deploy.md).

## TODO

- Data collection and evaluation are now only supported on the GelSight Mini sensor. We will add support for ViTai GF225 and XenseWS in the near future.

## 👍 Citations
If you find our work useful, please consider citing:

```
@article{chen2026univtac,
  title={UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking},
  author={Chen, Baijun and Wan, Weijie and Chen, Tianxing and Guo, Xianda and Xu, Congsheng and Qi, Yuanyang and Zhang, Haojie and Wu, Longyan and Xu, Tianling and Li, Zixuan and others},
  journal={arXiv preprint arXiv:2602.10093},
  year={2026}
}
```

## 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

## Contact
<div style="text-align: center;">
  <img src="https://box.nju.edu.cn/seafhttp/f/fc1021a908ff49309f22/?op=view" alt="Wechat Group" width="300"/>
</div>