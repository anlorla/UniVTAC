import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy

import os
import cv2
import yaml
import numpy as np
import torch
from .act_policy import ACT
# from act_policy import ACT
from torchvision import transforms

class Policy(BasePolicy):
# class Policy:
    def __init__(self, args):
        """Initialize ACT policy for TacArena deployment"""
        self.train_config_name = os.environ.get('TRAIN_CONFIG', 'train_config')
        self.ep_num = os.environ.get('EP_NUM', '50')
        self.task_name = args['task_name']
        self.is_dual = args['task_name'].startswith('dual_') or 'dual' in self.train_config_name

        with open(Path(__file__).parent / f'{self.train_config_name}.yml', 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)

        default_ckpt_dir = (
            Path(__file__).parent
            / "act_ckpt"
            / f"act-{args['task_name']}"
            / f"{args['task_config']}-{self.ep_num}"
            / self.train_config_name
        )
        ckpt_dir = Path(os.environ.get('CKPT_DIR', default_ckpt_dir)).expanduser()

        self.camera_names = train_config.get('camera_names', ['cam_high'])
        self.tactile_names = train_config.get('tactile_names', ['tac_left', 'tac_right'])
        if self.is_dual:
            self.camera_type = 'all'
            print(
                f"Using dual-arm ACT inputs cameras={self.camera_names} "
                f"tactiles={self.tactile_names}"
            )
        else:
            with open(Path(__file__).parent.parent / 'task_settings.json', 'r') as f:
                task_settings = json.load(f)
            assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
            self.camera_type = task_settings[self.task_name].get('camera_type', 'head')
            print(f"Using camera type '{self.camera_type}' for task '{self.task_name}'")
        
        train_config.update({
            'task_name': f"sim-{args['task_name']}-{args['task_config']}-{self.ep_num}",
            'task_config': args['task_config'],
            'ckpt_dir': str(ckpt_dir),
            "seed": args.get('seed', 0),
            "num_epochs": 1
        })
        
        # Initialize ACT model (RoboTwin_Config=None for TacArena)
        self.model = ACT(train_config)
        print(f"ACT policy loaded from {ckpt_dir}")

    @staticmethod
    def _camera_transform(img: torch.Tensor):
        img = transforms.Resize((256, 256))(img.permute(2, 0, 1))  # HWC -> CHW
        img = img / 255.0
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img

    @staticmethod
    def _tactile_transform(img: torch.Tensor):
        img = transforms.Resize((256, 256))(img.permute(2, 0, 1))  # HWC -> CHW
        img = img / 255.0
        return img

    @staticmethod
    def _joint8(joint):
        if isinstance(joint, torch.Tensor):
            return joint[:8].detach().cpu().numpy()
        return np.asarray(joint[:8])

    @staticmethod
    def _get_tactile_rgb_marker(observation, name):
        tactile = observation.get("tactile", {})
        aliases = {
            "left_gsmini": "left_tactile",
            "right_gsmini": "right_tactile",
        }
        key = name if name in tactile else aliases.get(name, name)
        if key not in tactile:
            raise KeyError(f"Missing tactile observation '{name}'/'{key}'")
        return tactile[key]["rgb_marker"]

    def _encode_dual_obs(self, observation):
        camera_map = {
            "cam_high": "head",
            "cam_wrist": "wrist",
            "cam_wrist_b": "wrist_b",
        }
        tactile_map = {
            "tac_left": "left_tactile",
            "tac_right": "right_tactile",
            "tac_left_b": "left_tactile_b",
            "tac_right_b": "right_tactile_b",
        }

        ret = {}
        for dst_name in self.camera_names:
            src_name = camera_map[dst_name]
            ret[dst_name] = self._camera_transform(observation["observation"][src_name]["rgb"])

        for dst_name in self.tactile_names:
            src_name = tactile_map[dst_name]
            ret[dst_name] = self._tactile_transform(
                self._get_tactile_rgb_marker(observation, src_name)
            )

        qpos_a = self._joint8(observation["embodiment"]["joint"])
        qpos_b = self._joint8(observation["embodiment_b"]["joint"])
        ret["qpos"] = np.concatenate([qpos_a, qpos_b], axis=0)
        return ret

    def encode_obs(self, observation):
        """
        Encode TacArena observation to ACT input format
        
        Input (TacArena):
            observation = {
                "observation": {"head": {"rgb": torch.Tensor([H, W, 3])}},  # HWC, 0-255
                "joint_action": torch.Tensor([9])  # [arm(7), gripper(1), extra(1)]
            }
            camera: 480x270
            tactile: 320x240
        
        Output (ACT):
            obs = {
                "qpos": torch.Tensor([8])  # [arm(7), gripper(1)]
                "cam_high": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_left": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_right": torch.Tensor([3, 256, 256]),  # CHW, 0-1
            }
        """
        if self.is_dual:
            return self._encode_dual_obs(observation)

        if self.camera_type == 'all':
            cam_high = self._camera_transform(observation["observation"]["head"]["rgb"])
            cam_wrist = self._camera_transform(observation["observation"]["wrist"]["rgb"])
        else:
            cam_high = self._camera_transform(observation["observation"][self.camera_type]["rgb"])

        left_tac = self._tactile_transform(
            self._get_tactile_rgb_marker(observation, "left_gsmini")
        )
        right_tac = self._tactile_transform(
            self._get_tactile_rgb_marker(observation, "right_gsmini")
        )
        
        # Extract joint positions (8D: 7 arm + 1 gripper)
        qpos = self._joint8(observation["embodiment"]["joint"])

        ret = {
            "cam_high": cam_high,
            "tac_left": left_tac,
            "tac_right": right_tac,
            "qpos": qpos,
        }
        if self.camera_type == 'all':
            ret["cam_wrist"] = cam_wrist
        return ret

    def _dual_take_action(self, task, action: torch.Tensor):
        if task.take_action_cnt >= task.cfg.step_lim or task.eval_success:
            return True, task.eval_success
        if action.numel() < 16:
            raise ValueError(f"Dual-arm ACT action must be 16D, got {action.numel()}")

        task.take_action_cnt += 1
        task.logger.info(f"step: {task.take_action_cnt} / {task.cfg.step_lim}")
        task._robot_manager.set_arm(action[:7], force=True)
        task._robot_manager.set_gripper(action[7], force=True)
        task._robot_manager_b.set_arm(action[8:15], force=True)
        task._robot_manager_b.set_gripper(action[15], force=True)
        task._step()

        if task.check_success():
            task.eval_success = True
        return True, task.eval_success

    def eval(self, task, observation):
        """
        Evaluate ACT policy on TacArena task
        
        Args:
            task: TacArena BaseTask instance
            observation: Current observation from environment
        """
        
        # Get action from ACT model (returns (1, 8) numpy array)
        obs = self.encode_obs(observation)
        if self.model.t % 10 == 0:
            self.save(task.get_frame_shot(observation), task.take_action_cnt)
        action = self.model.get_action(obs).reshape(-1)
        action = torch.from_numpy(action).to(task.device).float()
        if self.is_dual:
            exec_succ, eval_succ = self._dual_take_action(task, action)
        else:
            exec_succ, eval_succ = task.take_action(action, action_type='qpos')

    def reset(self):
        """Reset ACT model state (temporal aggregation and timestep counter)"""
        if hasattr(self.model, 'reset'):
            self.model.reset()

    def save(self, img, t):
        from PIL import Image
        from PIL import ImageDraw, ImageFont
        
        obs = Image.fromarray(img.cpu().numpy())

        draw = ImageDraw.Draw(obs)
        font = ImageFont.load_default()

        draw.text((obs.width-100, obs.height-60), f'{t:03d}', fill=(255, 0, 0), font=font)
        obs.save(f'ACT_{self.task_name}_{self.train_config_name}.png')
