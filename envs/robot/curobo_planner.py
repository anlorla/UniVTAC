from curobo.geom.transform import pose_multiply
import os
import numpy as np
import transforms3d as t3d
from curobo.types.robot import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.types.math import Pose as CuroboPose
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Mesh
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
import torch
import yaml
from curobo.util import logger
from copy import deepcopy

from pydantic import constr
logger.setup_logger(level="error", logger_name="curobo")

from pathlib import Path
from ..utils.transforms import *
from isaaclab.utils import configclass

@configclass
class CuroboPlannerCfg:
    dt: float = 1/120
    yaml_path: str = None
    robot_prime_path: str = "/World/robot"

    all_joints_name: list[str] = None
    active_joints_name: list[str] = None

    time_dilation_factor: float = 1.0

class CuroboPlanner:
    def __init__(
        self,
        task: 'BaseTask',
        cfg: CuroboPlannerCfg,
        robot_origin_pose:Pose,
    ):
        super().__init__()
        logger.setup_logger(level="error", logger_name="'curobo")

        self.cfg = cfg
        self.task = task
        self.dt = cfg.dt
        self.robot_prime_path = cfg.robot_prime_path
        self.robot_origin_pose = robot_origin_pose
        self.active_joints_name = cfg.active_joints_name
        self.all_joints = cfg.all_joints_name
        # translate from baselink to arm's base
        with open(self.cfg.yaml_path, "r") as f:
            yml_data = yaml.safe_load(f)

        file_dir = Path(self.cfg.yaml_path).parent
        urdf_path = yml_data['robot_cfg']['kinematics']['urdf_path']
        if not Path(urdf_path).is_absolute():
            yml_data['robot_cfg']['kinematics']['urdf_path'] = str(file_dir / urdf_path)
        collision_spheres = yml_data['robot_cfg']['kinematics']['collision_spheres']
        if not Path(collision_spheres).is_absolute():
            yml_data['robot_cfg']['kinematics']['collision_spheres'] = str(file_dir / collision_spheres)

        self.frame_bias = yml_data["planner"]["frame_bias"]

        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(self.task.scene.stage)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg=yml_data,
            world_model=self.get_curr_world_cfg(),
            interpolation_dt=self.dt,
            position_threshold=0.001,
            rotation_threshold=0.01,
            high_precision=True,
            collision_checker_type=CollisionCheckerType.MESH,
            # 原值 0.4m(40cm)是 collision cost 的软激活距离, 远超 cuRobo 默认 0.025m。
            # 桌面(ground_plate)被纳入障碍后, 抓取目标(离桌面~13cm)整个落在 40cm 软碰撞带内,
            # 任何接近都被判为碰撞 -> trajopt 不收敛(pos_err~0.9) -> DT_EXCEPTION/规划失败。
            # 抓取本就需要贴近物体, 必须用小激活距离; 0.025m 为 cuRobo 默认且实测可正常抓取。
            collision_activation_distance=0.025,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
    
    def reset(self):
        self.motion_gen.reset()

    def get_curr_world_cfg(self):
        # 障碍物参考系 = 本臂自己的基座(env_0 实例)。由 robot_prime_path 推出,
        # 不再写死, 这样第二条臂(/Robot_b)也能用各自基座系取障碍。
        ref_prim_path = self.robot_prime_path.replace('env_.*', 'env_0')
        obstacles = self.usd_helper.get_obstacles_from_stage(
            reference_prim_path=ref_prim_path,
            only_paths=[
                '/World/envs/env_0/ground_plate'
            ],
        ).get_collision_check_world()

        # 该忽略的 actor(随夹爪一起动的在手件, 或本臂要穿过的目标), 不作为静态障碍物,
        # 否则长 peg 盖住夹爪/小孔被当实心 -> 任何运动都判碰撞 -> 规划失败。
        # 双臂时各臂忽略集不同(owning_manager.ignore_actors), 与任务级集合并集(单臂向后兼容)。
        ignore = set(getattr(self.task, 'planner_ignore_actors', None) or set())
        owning = getattr(self, 'owning_manager', None)
        if owning is not None:
            ignore |= set(getattr(owning, 'ignore_actors', None) or set())
        for name, actor in self.task._actor_manager.actors.items():
            if name in ignore:
                continue
            mesh = Mesh.from_pointcloud(actor.vertices, pitch=0.005, name=name)
            obstacles.add_obstacle(mesh)
        return obstacles
 
    def update_world(self):
        self.motion_gen.update_world(self.get_curr_world_cfg())

    def plan_path(
        self,
        curr_joint_pos: torch.Tensor,
        curr_joint_vel: torch.Tensor,
        target_ee_pose,
        real_robot_pose,
        pre_dis=None,
        constraint_pose=None,
        time_dilation_factor=None
    ):
        # self.update_world()
        target_pose = calculate_target_pose(
            real_robot_pose, self.robot_origin_pose, target_ee_pose)
        # transformation from world to arm's base
        target_pose = target_pose.rebase(to_coord=self.robot_origin_pose).add_bias(
            self.frame_bias, coord='world', clone=False
        )
        goal_pose_of_ee = CuroboPose.from_list(target_pose.tolist())
        joint_indices = np.array([
            self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints])
        joint_pos = curr_joint_pos[joint_indices].reshape(1, -1)
        joint_vel = curr_joint_vel[joint_indices].reshape(1, -1)
        
        start_joint_states = JointState(
            position=joint_pos,
            velocity=joint_vel,
            acceleration=torch.zeros_like(joint_pos),
            jerk=torch.zeros_like(joint_pos),
            joint_names=self.active_joints_name,
        )
        # plan
        if time_dilation_factor is None:
            time_dilation_factor = self.cfg.time_dilation_factor
        plan_config = MotionGenPlanConfig(max_attempts=10, time_dilation_factor=time_dilation_factor)

        pose_cost_metric = None
        if constraint_pose is not None:
            if pre_dis is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                    offset_position=self.motion_gen.tensor_args.to_device([0.0, 0.0, pre_dis])
                )
            else:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose)
                )
        elif pre_dis is not None and pre_dis != 0.0:
            pose_cost_metric = PoseCostMetric.create_grasp_approach_metric(
                offset_position=pre_dis, tstep_fraction=0.6, linear_axis=2)

        if pose_cost_metric is not None:
            plan_config.pose_cost_metric = pose_cost_metric

        return self.motion_gen.plan_single(
            start_joint_states, goal_pose_of_ee, plan_config)

    def solve_ik(self, target_ee_pose, real_robot_pose, curr_joint_pos=None):
        # direct single-shot IK (no trajectory planning) -- fast, avoids curobo plan_single GPU contention
        target_pose = calculate_target_pose(
            real_robot_pose, self.robot_origin_pose, target_ee_pose)
        target_pose = target_pose.rebase(to_coord=self.robot_origin_pose).add_bias(
            self.frame_bias, coord='world', clone=False)
        goal_pose_of_ee = CuroboPose.from_list(target_pose.tolist())
        retract = None
        if curr_joint_pos is not None:
            joint_indices = np.array([
                self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints])
            retract = curr_joint_pos[joint_indices].reshape(1, -1)
        # UNIVTAC_IK_SEEDCFG=1: seed IK from the CURRENT joints (single seed) so the
        # solver converges to the nearest solution branch, avoiding null-space wrist
        # flips (J5/J7 jumping up to ~2 rad between frames at an identical EE pose).
        # EE accuracy is unchanged -- removes joint jitter. DEFAULT ON (set UNIVTAC_IK_SEEDCFG=0 to disable).
        seed_config = None
        if retract is not None and os.environ.get('UNIVTAC_IK_SEEDCFG', '1') == '1':
            seed_config = retract.view(1, 1, -1)
        result = self.motion_gen.solve_ik(
            goal_pose_of_ee, retract_config=retract, seed_config=seed_config)
        if bool(result.success.view(-1)[0].item()):
            return {'status': 'Success', 'position': result.js_solution.position.detach().reshape(-1)}
        return {'status': 'Fail', 'position': None}