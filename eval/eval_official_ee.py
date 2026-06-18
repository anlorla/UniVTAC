#!/usr/bin/env python3
"""Standalone EE eval — COPIES official scripts/eval_policy.py logic VERBATIM.

与官方 harness 的差异仅两处（均为必要适配）：
  1. WebsocketClientPolicy 连接在 Isaac AppLauncher **之前**建立
     （websockets.sync.connect 在 Isaac sim 起来后调用会直接阻塞死锁；
      eval_lingbot_delta 也是先连后启 sim）。
  2. 动作经 take_action(ee, action_type='ee') 执行（官方 openpi 用 'qpos'）。
其余全部与官方一致：expert_check 资格赛(play_once)、reset+instructions、
_get_observations、eval 循环、check_success/eval_success/check_early_stop、seeds.json。
"""
import sys
import os
import time
import json
import yaml
import argparse
import traceback
from pathlib import Path
from typing import Literal

sys.path.append(".")
sys.path.append("./policy")

import argparse as _ap
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval Policy (EE, standalone, official-aligned)")
parser.add_argument("task_name", type=str)
parser.add_argument("task_config", type=str)
parser.add_argument("--vta_host", default="192.168.50.18")
parser.add_argument("--vta_port", type=int, default=29542)
parser.add_argument("--prompt", default="Rotate and pull out the key")
parser.add_argument("--exec_horizon", type=int, default=0)
parser.add_argument("--expert_check", action="store_true")
parser.add_argument("--start_seed", type=int, default=-1)
parser.add_argument("--max_seed", type=int, default=-1)
parser.add_argument("--total_num", type=int, default=20)
parser.add_argument("--print_only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.livestream = 2
args_cli.num_envs = 1

# ============ STEP 1: 连 VTA server（必须在 Isaac 之前，否则同步 ws connect 死锁） ============
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.expanduser("~/yifan/vta_bridge"))
from remote_infer_lib.websocket_client_policy import WebsocketClientPolicy

print(f"[ee-eval] connecting VTA server {args_cli.vta_host}:{args_cli.vta_port} (before Isaac) ...", flush=True)
client = WebsocketClientPolicy(host=str(args_cli.vta_host), port=int(args_cli.vta_port))
print(f"[ee-eval] connected: {client.get_server_metadata()}", flush=True)

# ============ STEP 2: 启 Isaac ============
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import importlib

# -------- VTA ee helpers（与 eval_lingbot_delta 一致） --------
def to_u8(t):
    x = t[..., :3]
    x = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    x = np.squeeze(x)
    if x.dtype != np.uint8:
        x = (x * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(x)

def ee_to_rot6d_state(task):
    pose = task._robot_manager.get_ee_pose()
    p_xyz = np.asarray(pose.p, dtype=np.float64).reshape(-1)[:3]
    q_wxyz = np.asarray(pose.q, dtype=np.float64).reshape(-1)
    Rm = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    r6 = Rm[:, :2].T.reshape(-1)
    grip = float(task._robot_manager.get_gripper_qpos())
    s = np.zeros(20, dtype=np.float32); s[:3] = p_xyz; s[3:9] = r6; s[9] = grip
    return s

def rot6d10_to_ee(d):
    xyz = np.asarray(d[:3], dtype=np.float64)
    c1 = np.asarray(d[3:6], dtype=np.float64); c2 = np.asarray(d[6:9], dtype=np.float64)
    b1 = c1 / (np.linalg.norm(c1) + 1e-8)
    c2 = c2 - (b1 @ c2) * b1; b2 = c2 / (np.linalg.norm(c2) + 1e-8)
    b3 = np.cross(b1, b2)
    Rm = np.stack([b1, b2, b3], axis=1)
    q_xyzw = R.from_matrix(Rm).as_quat()
    grip = float(d[9])
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]),
            float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), grip]

_patched = {"done": False}
def _ensure_gripper_patch(task):
    if _patched["done"]:
        return
    _rm = task._robot_manager; _o = _rm.plan_gripper
    def _pg(pos, type='percent', _o=_o):
        if isinstance(pos, torch.Tensor):
            pos = float(pos.reshape(-1)[0])
        elif hasattr(pos, '__len__'):
            pos = float(pos[0])
        else:
            pos = float(pos)
        return _o(pos, type)
    _rm.plan_gripper = _pg
    _patched["done"] = True

def vta_eval(task, observation):
    """对应官方 policy.eval：唯一区别 take_action 用 'ee' 不用 'qpos'。"""
    _ensure_gripper_patch(task)
    cam = observation["observation"]; tac = observation["tactile"]
    obs = {
        'obs': {
            'observation.images.top': to_u8(cam["head"]["rgb"]),
            'observation.images.wrist_l': to_u8(cam["wrist"]["rgb"]),
        },
        'tactile': {
            'observation.images.tactile_a': to_u8(tac["left_tactile"]["rgb"]),
            'observation.images.tactile_b': to_u8(tac["right_tactile"]["rgb"]),
        },
        'current_state': ee_to_rot6d_state(task).tolist(),
        'prompt': args_cli.prompt,
    }
    ret = client.infer(obs)
    action = np.asarray(ret['action'])
    steps = np.transpose(action, (1, 2, 0)).reshape(-1, action.shape[0])
    for i, d in enumerate(steps):
        if args_cli.exec_horizon and i >= args_cli.exec_horizon:
            break
        ee = torch.tensor(rot6d10_to_ee(d[:10]), dtype=torch.float32)
        task.take_action(ee, action_type='ee')   # ★ 官方为 'qpos'
        if getattr(task, "eval_success", False):
            return
        if task.take_action_cnt >= task.cfg.step_lim:
            return

# -------- log（拷自官方） --------
log_path = Path('./log')
def log(msg):
    global log_path
    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    if not args_cli.print_only:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
    print(msg, flush=True)

# -------- eval_policy()：逐字拷自官方，仅 policy.eval→vta_eval、policy.reset→client reset --------
def eval_policy(task, expert_check, start_seed, max_seed, test_total_num, instructions, instruciton_type='seen'):
    from shutil import ExecError
    test_num, succ_num, seed = 0, 0, start_seed
    seed_path = task.save_root.parent / 'seeds.json'
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    if seed_path.exists():
        with open(seed_path, 'r') as f:
            seed_status = json.load(f)
    else:
        seed_status = {}

    while test_num < test_total_num and (max_seed == -1 or seed <= max_seed):
        if not seed_status.get(str(seed), True):
            seed += 1
            continue

        if expert_check and str(seed) not in seed_status:
            test_start = time.perf_counter()
            task.mode = 'eval_test'
            try:
                task.reset(seed=seed)
                task.play_once()
                if not task.check_success() or not task.plan_success:
                    raise ExecError(f'seed {seed} Expert check failed, check {task.check_success()}, plan {task.plan_success}.')
                else:
                    seed_status[seed] = True
                    with open(seed_path, 'w') as f:
                        json.dump(seed_status, f)
                test_cost = time.perf_counter() - test_start
                task.clean_cache(result='test_success')
                log(f'Expert check succ, seed {seed}, cost {test_cost:.2f}s')
            except Exception as e:
                test_cost = time.perf_counter() - test_start
                log(f'Expert check failed, seed {seed}, cost {test_cost:.2f}s, with exception {e}')
                task.clean_cache(result='test_fail')
                seed_status[seed] = False
                with open(seed_path, 'w') as f:
                    json.dump(seed_status, f)
                seed += 1
                continue
        test_num += 1

        succ = False
        eval_start = time.perf_counter()
        task.mode = 'eval'
        try:
            task.reset(seed=seed, instructions=instructions[instruciton_type])
            task.mean_steps = task.cfg.step_lim
            try:
                client.infer(dict(reset=True, prompt=args_cli.prompt))   # = policy.reset()
            except Exception:
                pass
            while task.take_action_cnt < task.cfg.step_lim:
                observation = task._get_observations()
                vta_eval(task, observation)                              # = policy.eval()
                if task.eval_success:
                    succ = True
                    break
                if task.check_early_stop():
                    break
        except Exception as e:
            log(f"[{test_num:<3d}] Seed {seed} occurred exception: {e}\n{traceback.format_exc()}")
            succ_status = 'error'
            task.clean_cache(result=succ_status)
            test_num -= 1
        else:
            eval_cost = time.perf_counter() - eval_start
            if succ:
                succ_num += 1
            succ_status = 'success' if succ else 'failed'
            task.clean_cache(result=succ_status)
            log(f"[{test_num:<3d}] Seed {seed} {succ_status} after {eval_cost:.2f} s.\n"
                f"steps: {task.step_count:<5d}, actions: {task.take_action_cnt:<5d}.\n"
                f"Instruction: {task.instruction}\n"
                f"Total {succ_num}/{test_num}({succ_num/max(1,test_num)*100:.2f}%) success.")
        finally:
            seed += 1

    return {'test_num': test_num, 'succ_num': succ_num}

def get_config(file, default_root, type):
    if type == 'yaml':
        file = Path(file) if (file.endswith('.yml') or file.endswith('.yaml')) else default_root / f'{file}.yml'
        with open(file, 'r') as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader), file
    else:
        file = Path(file) if file.endswith('.json') else default_root / f'{file}.json'
        with open(file, 'r') as f:
            return json.load(f), file

def main():
    global log_path
    task_file_name = args_cli.task_name
    task_config, task_config_file = get_config(
        args_cli.task_config, default_root=Path('./task_config'), type='yaml')

    # instructions：同官方，从 instructions/{task}.json 取；没有则 Empty
    try:
        instructions, _ = get_config(task_file_name, default_root=Path('./instructions'), type='json')
    except Exception:
        instructions = {'seen': ['Empty'], 'unseen': ['Empty']}

    task_module = importlib.import_module(f"envs.{task_file_name}")
    curr_time = time.strftime(r'%Y-%m-%d_%H:%M:%S')

    env_cfg = task_module.TaskCfg()
    env_cfg.save_dir = Path('eval_result') / 'lingbot_vta_ee' / task_file_name / task_config_file.stem / curr_time
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    import os as _os

    if _os.environ.get('EVAL_STEP_LIM'):

        env_cfg.step_lim = int(_os.environ['EVAL_STEP_LIM'])

    task = task_module.Task(env_cfg, mode='eval')
    log_path = task.save_root / "log.log"
    log(f"Task: {task_file_name} | config: {task_config_file.stem} | server {args_cli.vta_host}:{args_cli.vta_port}")
    log(f"prompt='{args_cli.prompt}' exec_horizon={args_cli.exec_horizon} expert_check={args_cli.expert_check}")

    start_seed = 1000000 if args_cli.start_seed == -1 else args_cli.start_seed
    results = eval_policy(
        task=task, expert_check=args_cli.expert_check,
        start_seed=start_seed, max_seed=args_cli.max_seed,
        test_total_num=args_cli.total_num, instructions=instructions,
        instruciton_type='seen')
    log(f"Final Result: {results['succ_num']}/{results['test_num']}"
        f"({results['succ_num']/max(1,results['test_num'])*100:.2f}%) success.")

    task.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
