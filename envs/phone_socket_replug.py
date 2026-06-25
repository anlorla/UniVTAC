from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Phone Socket Replug
#   单臂任务: 从 source base 里拔出 phone plug -> 搬运到 target base 上方 -> 重新插入。
#   plug: PHONE_PLUG.usd
#   base: PHONE_SOCKET_BASE.usd (source/target 共用同一资产)
# ----------------------------------------------------------------------------
#   几何约定:
#   - plug 的 charging tongue 朝 local -Z
#   - base 的孔口在顶面, 插入方向沿 world -Z
#   - reset 时 plug 已经插在 source base 里, 机器人需要先抓住机身再拔出
# ============================================================================

BASE_HEIGHT = 0.012
HOLE_DEPTH = 0.007
PLUG_TIP = 0.01675
GRASP_DZ = 0.004
UNPLUG_LIFT = 0.10
INSERT_DEPTH = 0.011
UPRIGHT = [1, 0, 0, 0]

SOURCE_POS = Pose([0.45, -0.10, 0.002], UPRIGHT)
TARGET_POS = Pose([0.58, 0.12, 0.002], UPRIGHT)


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.88, 0.0, 0.20), rot=(0.611, 0.389, 0.370, 0.581), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
            ),
            width=480,
            height=270,
            update_period=1 / 120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480,
            height=270,
            update_period=1 / 120,
        ),
    ]
    step_lim = 900


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.0
        cfg.sim.physics_material.static_friction = 2.0
        cfg.uipc_sim.contact.default_friction_ratio = 2.0
        self.planner_ignore_actors = {"plug"}
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        self.source = self._actor_manager.add_from_usd_file(
            name="source", asset_path="PHONE_SOCKET_BASE.usd", pose=SOURCE_POS, density=1e5
        )
        self.target = self._actor_manager.add_from_usd_file(
            name="target", asset_path="PHONE_SOCKET_BASE.usd", pose=TARGET_POS, density=1e5
        )
        self.plug = self._actor_manager.add_from_usd_file(
            name="plug", asset_path="PHONE_PLUG.usd", pose=Pose([0.45, 0.0, 0.25], UPRIGHT), density=3e3
        )

    def _reset_actors(self):
        source_pose = SOURCE_POS.add_offset(self.create_noise([0.004, 0.004, 0.0]))
        target_pose = TARGET_POS.add_offset(self.create_noise([0.004, 0.004, 0.0]))
        self.source.set_pose(source_pose)
        self.target.set_pose(target_pose)

        self.source_hole_pose = self.source.get_pose().add_bias([0.0, 0.0, BASE_HEIGHT + PLUG_TIP])
        self.target_hole_pose = self.target.get_pose().add_bias([0.0, 0.0, BASE_HEIGHT + PLUG_TIP])
        plug_pose = self.source_hole_pose.add_bias([0.0, 0.0, -HOLE_DEPTH * 0.85])
        self.plug.set_pose(plug_pose)

        self.metadata["source_xy"] = [float(source_pose.p[0]), float(source_pose.p[1])]
        self.metadata["target_xy"] = [float(target_pose.p[0]), float(target_pose.p[1])]

    def _close_gripper_direct(self, percent=0.0, settle_steps=10, is_save=True):
        rm = self._robot_manager
        target = rm.gripper_percent2qpos(percent)
        pos = torch.tensor([target, target], device=rm.device, dtype=rm.robot.data.joint_pos.dtype)
        self.atom_id += 1
        self.atom_tag = "close_direct"
        rm.set_gripper(pos, force=True)
        for _ in range(settle_steps):
            rm.set_gripper(pos, force=True)
            self._step(is_save=is_save)
        self._update_render()

    def _grasp_plug_from_source(self):
        self.move(self.atom.open_gripper(1.0))
        plug_pose = self.plug.get_pose()
        gp = np.array([plug_pose.p[0], plug_pose.p[1], plug_pose.p[2] + GRASP_DZ], dtype=float)
        gc = construct_grasp_pose(gp, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        gc_high = gc.add_bias([0.0, 0.0, 0.06], coord="world")
        ee_high = self._robot_manager.gripper_center_to_ee(gc_high)
        ee_grasp = self._robot_manager.gripper_center_to_ee(gc)
        self.move(self.atom.move_to_pose(ee_high), tag="pre_grasp")
        self.move(self.atom.move_to_pose(ee_grasp), tag="grasp_drop")
        self.move(self.atom.close_gripper(0.0), tag="grasp_close")
        self._close_gripper_direct(0.0, settle_steps=12, is_save=True)

    def pre_move(self):
        self.delay(12)

    def _play_once(self):
        self._grasp_plug_from_source()
        self.move(
            self.atom.move_by_displacement(z=UNPLUG_LIFT, xyz_coord="world"),
            tag="unplug_lift",
            constraint_pose=[1, 1, 1, 1, 1, 0],
            time_dilation_factor=0.5,
        )

        target_hover = self.target_hole_pose.add_bias([0.0, 0.0, 0.06])
        self.move(
            self.atom.place_actor(
                self.plug,
                target_pose=target_hover,
                pre_dis=0.03,
                dis=0.01,
                is_open=False,
            ),
            tag="move_over_target",
            time_dilation_factor=0.5,
        )
        self.move(
            self.atom.place_actor(
                self.plug,
                target_pose=self.target_hole_pose,
                pre_dis=0.015,
                dis=0.003,
                is_open=False,
            ),
            tag="align_target",
            time_dilation_factor=0.5,
        )
        self.move(
            self.atom.move_by_displacement(z=INSERT_DEPTH * 0.65, xyz_coord="local"),
            tag="insert_stage1",
            constraint_pose=[1, 1, 1, 1, 1, 0],
            time_dilation_factor=0.5,
        )
        self.move(
            self.atom.move_by_displacement(z=INSERT_DEPTH * 0.35, xyz_coord="local"),
            tag="insert_stage2",
            constraint_pose=[1, 1, 1, 1, 1, 0],
            time_dilation_factor=0.5,
        )
        self.move(self.atom.open_gripper(1.0), tag="release")
        self.delay(20, is_save=True)

    def check_success(self):
        rel_target = self.plug.get_pose().rebase(self.target_hole_pose)
        rel_source = self.plug.get_pose().rebase(self.source_hole_pose)
        self.metadata["rel_target"] = rel_target.tolist()
        self.metadata["rel_source"] = rel_source.tolist()
        inserted_target = (
            np.all(np.abs(rel_target.p[:2]) < np.array([0.005, 0.005]))
            and rel_target.p[2] < 0.006
            and np.dot(rel_target.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.95
        )
        clear_source = abs(rel_source.p[2]) > 0.012 or np.linalg.norm(np.array(rel_source.p[:2])) > 0.02
        return bool(inserted_target and clear_source)
