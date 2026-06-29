from ._base_task import *
import numpy as np
import torch
from pxr import UsdGeom

# ============================================================================
# Dual Cup HANDOVER-PLACE —— 双臂杯子交接并放置:
#   初始: 一个纸杯放在左臂一侧桌面, 杯口朝 local +Z, 闭底朝 local -Z。
#   流程: A 从上方夹取杯子侧壁 -> 移到双臂中间偏右的交接位 ->
#         B 从另一侧夹住同一个杯子 -> A 松开并撤离 -> B 将杯子放到右侧目标位。
#     cup : assets/objects/CUP.usd
#       - 源表面网格: assets/objects/CUP.obj
#       - USD 内嵌 tet 四面体网格属性: tet_points / tet_indices /
#         tet_surf_points / tet_surf_indices
#       - 生成脚本: scripts/asset_tools/build_cup.py,
#         scripts/asset_tools/bake_tet.py,
#         scripts/asset_tools/build_cup_usd.py
#   对象几何:
#     CUP.usd 是薄壁截锥纸杯, 杯口朝 local +Z, 闭底朝 local -Z。
#     原始尺寸约口外径 76mm、底外径 54mm、高 92mm, 原点在杯体中心,
#     local z 范围约 [-46mm, +46mm]。
# ============================================================================

# ---- 几何 (m) ----
CUP_SCALE = 0.92
CUP_HALF = 0.046 * CUP_SCALE
CUP_TOP_R = 0.038 * CUP_SCALE
TABLE_TOP = 0.004
PLATE_HALF_Z = 0.004
GAP = 0.006

# ---- 摆位 (世界系) ----
UPRIGHT = [1, 0, 0, 0]
CUP_START = Pose([0.40, 0.22, TABLE_TOP + CUP_HALF + GAP], UPRIGHT)
HANDOVER_TARGET = Pose([0.50, 0.02, 0.17], UPRIGHT)
TARGET_PLATE_POS = Pose([0.50, -0.26, TABLE_TOP + PLATE_HALF_Z], UPRIGHT)
PLACE_TARGET = Pose([TARGET_PLATE_POS.p[0], TARGET_PLATE_POS.p[1], TABLE_TOP + 2 * PLATE_HALF_Z + CUP_HALF + GAP], UPRIGHT)
PLANNER_ANCHOR_POS = Pose([0.82, 0.38, TABLE_TOP + PLATE_HALF_Z], UPRIGHT)

# ---- 动作参数 ----
GRASP_R = CUP_TOP_R + 0.002
SIDE_GRASP_R = 0.0
GRASP_DZ = CUP_HALF - 0.011
SIDE_GRASP_DZ = 0.0
GRASP_PRE_DIS = 0.12
SIDE_GRASP_PRE_DIS = 0.08
PRE_GRIPPER_OPEN = 0.62
SIDE_PRE_GRIPPER_OPEN = 0.82
SIDE_CLOSE_GRIPPER_POS = 0.70
HANDOVER_HOVER = 0.08
PLACE_HOVER = 0.11
RETRACT_Z = 0.10
ARM_A_PARK_POS = [0.36, 0.30, 0.24]
ARM_B_RETRACT_POS = [0.46, -0.32, 0.24]
ARM_B_SIDE_STAGE_POS = [0.50, -0.20, 0.28]
B_RELEASE_OPEN = 0.82
B_RELEASE_SIDE_RETREAT = 0.055


@configclass
class TaskCfg(BaseTaskCfg):
    dual_arm = True
    use_adaptive_grasp = False
    viewer = ViewerCfg(eye=(1.45, 0.0, 0.95), lookat=(0.45, 0.0, 0.12))
    video_size = (1760, 320)
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SCENE_ASSETS_ROOT / "plate.usda"),
            scale=(2.0, 2.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
            ),
        ),
    )
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(1.35, 0.0, 0.85),
                rot=(0.6352, 0.3107, 0.3107, 0.6352),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5,
                focus_distance=1.0,
                horizontal_aperture=3.6,
                clipping_range=(0.1, 100.0),
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
        CameraCfg(
            name="wrist_b",
            prim_path="/World/envs/env_.*/Robot_b/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480,
            height=270,
            update_period=1 / 120,
        ),
    ]
    step_lim = 1500
    reset_time_limit = 1200.0


class Task(BaseTask):
    def __init__(
        self,
        cfg: BaseTaskCfg,
        mode: Literal["collect", "eval"] = "collect",
        render_mode: str | None = None,
        **kwargs,
    ):
        cfg.sim.physics_material.dynamic_friction = 5.0
        cfg.sim.physics_material.static_friction = 5.0
        cfg.uipc_sim.contact.default_friction_ratio = 5.0
        self.planner_ignore_actors = {"cup", "target_plate"}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {"cup"}
        # B has to place the cup onto the plate, so the plate cannot be a B-side
        # collision obstacle during the descent. planner_anchor is hidden from the
        # viewer but kept as a harmless non-ignored actor so cuRobo does not get an
        # empty actor world.
        self._robot_manager_b.ignore_actors = {"cup", "target_plate"}

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        self.cup = self._actor_manager.add_from_usd_file(
            name="cup",
            asset_path="CUP.usd",
            pose=CUP_START,
            scale=(CUP_SCALE, CUP_SCALE, CUP_SCALE),
            density=2e5,
        )
        # 目标盘: B 接过杯子后把杯子放到盘子上。路径规划会忽略该盘,
        # 让最后下降像叠杯任务一样由物理接触/成功判定处理。
        self.target_plate = self._actor_manager.add_from_usd_file(
            name="target_plate",
            asset_path="PLATE.usd",
            pose=TARGET_PLATE_POS,
            density=1e4,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(kinematic=True),
        )
        self.planner_anchor = self._actor_manager.add_from_usd_file(
            name="planner_anchor",
            asset_path="PLATE.usd",
            pose=PLANNER_ANCHOR_POS,
            density=1e4,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(kinematic=True),
        )
        self._hide_actor("planner_anchor")

    def _hide_actor(self, name):
        prim = self.scene.stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()

    def _reset_actors(self):
        noise = self.create_noise([0.003, 0.003, 0.0])
        self.cup.set_pose(CUP_START.add_offset(noise))
        self.target_plate.set_pose(TARGET_PLATE_POS)
        self.planner_anchor.set_pose(PLANNER_ANCHOR_POS)
        self.metadata["handover_target"] = [float(v) for v in HANDOVER_TARGET.p]
        self.metadata["place_target"] = [float(v) for v in PLACE_TARGET.p]
        self.metadata["target_plate"] = [float(v) for v in TARGET_PLATE_POS.p]

    # ---------------------------------------------------------------- helpers
    def _register_rim_grasp(
        self,
        actor,
        rim_dir,
        grasp_from=(0.0, 0.0, 1.0),
        camera_up=(1.0, 0.0, 0.0),
        grasp_r=GRASP_R,
        grasp_dz=GRASP_DZ,
    ):
        rd = np.array(rim_dir, dtype=float)
        rd /= np.linalg.norm(rd)
        center = actor.get_pose()
        gp = np.array(center.p, dtype=float) + rd * grasp_r + np.array([0.0, 0.0, grasp_dz])
        grasp_pose = construct_grasp_pose(
            gp,
            np.array(grasp_from, dtype=float),
            np.array(camera_up, dtype=float),
        )
        return actor.register_point(grasp_pose, type="contact")

    def _grasp_cup(
        self,
        atom,
        rm,
        arm,
        rim_dir,
        weld=True,
        grasp_from=(0.0, 0.0, 1.0),
        camera_up=(1.0, 0.0, 0.0),
        grasp_r=GRASP_R,
        grasp_dz=GRASP_DZ,
        open_width=PRE_GRIPPER_OPEN,
        close_width=0.0,
        pre_dis=GRASP_PRE_DIS,
    ):
        self.move(atom.open_gripper(open_width), arm=arm)
        grasp_id = self._register_rim_grasp(
            self.cup,
            rim_dir,
            grasp_from=grasp_from,
            camera_up=camera_up,
            grasp_r=grasp_r,
            grasp_dz=grasp_dz,
        )
        self.move(
            atom.grasp_actor(
                self.cup,
                contact_point_id=grasp_id,
                pre_dis=pre_dis,
                dis=0.0,
                is_close=False,
            ),
            arm=arm,
            time_dilation_factor=0.5,
        )
        self.move(atom.close_gripper(close_width, depth_threshold=None), arm=arm)
        if weld:
            self._set_weld(self.cup, rm)
        self.delay(8, is_save=True)

    def _set_weld(self, actor, rm):
        self._unweld_actor(actor)
        self.weld_actor(actor, rm)

    def _unweld_actor(self, actor):
        self._welds = [w for w in self._welds if w[0] is not actor]

    def _place_inhand(self, rm, atom, target_pose, arm):
        inhand = self.cup.get_pose().rebase(rm.get_gripper_center_pose())
        gc_mat = np.array(target_pose.to_transformation_matrix()) @ np.linalg.inv(
            np.array(inhand.to_transformation_matrix())
        )
        ee = rm.gripper_center_to_ee(Pose.from_matrix(gc_mat))
        self.move(atom.move_to_pose(ee), arm=arm, time_dilation_factor=0.5)

    def _move_gripper_center(self, rm, atom, target_pos, arm):
        gc_now = rm.get_gripper_center_pose()
        gc_target = Pose(target_pos, gc_now.q)
        self.move(
            atom.move_to_pose(rm.gripper_center_to_ee(gc_target)),
            arm=arm,
            time_dilation_factor=0.5,
        )

    def _move_side_stage(self):
        stage_gc = construct_grasp_pose(
            np.array(ARM_B_SIDE_STAGE_POS, dtype=float),
            np.array([0.0, -1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        )
        self.move(
            self.atom_b.move_to_pose(self._robot_manager_b.gripper_center_to_ee(stage_gc)),
            arm="b",
            time_dilation_factor=0.5,
        )

    def _release_arm(
        self,
        atom,
        arm,
        open_width=0.45,
        retract=True,
        park_pos=None,
        side_retreat_y=0.0,
    ):
        self.move(atom.open_gripper(open_width), arm=arm)
        if side_retreat_y != 0.0:
            self.move(
                atom.move_by_displacement(y=side_retreat_y, xyz_coord="world"),
                arm=arm,
                time_dilation_factor=0.5,
            )
        if retract:
            self.move(
                atom.move_by_displacement(z=RETRACT_Z, xyz_coord="world"),
                arm=arm,
                time_dilation_factor=0.5,
            )
        if park_pos is not None:
            rm, _ = self._arm(arm)
            self._move_gripper_center(rm, atom, park_pos, arm)

    def _dbg(self, tag):
        p = self.cup.get_pose()
        print(f"[HANDOVER_PLACE] {tag}: cup=({p.p[0]:.3f},{p.p[1]:.3f},{p.p[2]:.3f})", flush=True)

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(15)
        self._dbg("reset settled")

    def _play_once(self):
        # A: 从桌上抓杯并移动到交接位。沿 +Y 杯壁抓取, 复用叠杯任务中稳定的侧壁抓法。
        self._grasp_cup(self.atom_a, self._robot_manager, "a", rim_dir=(0, 1, 0), weld=True)
        self._place_inhand(
            self._robot_manager,
            self.atom_a,
            HANDOVER_TARGET.add_bias([0.0, 0.0, HANDOVER_HOVER], coord="world"),
            "a",
        )
        self._place_inhand(self._robot_manager, self.atom_a, HANDOVER_TARGET, "a")
        self._dbg("A moved cup to handover")

        # B: 从 -Y 侧水平接杯。先到抬高的横向预备姿态, 再让 grasp_actor 直线靠近。
        # 侧抓时目标用杯子中心轴而不是杯壁边缘: 夹爪中心在杯轴线上, 两指闭合到杯壁。
        self._move_side_stage()
        self._grasp_cup(
            self.atom_b,
            self._robot_manager_b,
            "b",
            rim_dir=(0, -1, 0),
            weld=False,
            grasp_from=(0, -1, 0),
            camera_up=(0, 0, 1),
            grasp_r=SIDE_GRASP_R,
            grasp_dz=SIDE_GRASP_DZ,
            open_width=SIDE_PRE_GRIPPER_OPEN,
            close_width=SIDE_CLOSE_GRIPPER_POS,
            pre_dis=SIDE_GRASP_PRE_DIS,
        )
        self._set_weld(self.cup, self._robot_manager_b)
        self.delay(5, is_save=True)
        self._release_arm(self.atom_a, "a", retract=True, park_pos=ARM_A_PARK_POS)
        self._dbg("B received cup, A released")

        # B: 将杯子移动并放到右侧目标位。
        self._place_inhand(
            self._robot_manager_b,
            self.atom_b,
            PLACE_TARGET.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "b",
        )
        self._place_inhand(self._robot_manager_b, self.atom_b, PLACE_TARGET, "b")
        self._unweld_actor(self.cup)
        self.delay(5, is_save=True)
        self._release_arm(
            self.atom_b,
            "b",
            open_width=B_RELEASE_OPEN,
            retract=True,
            park_pos=ARM_B_RETRACT_POS,
            side_retreat_y=-B_RELEASE_SIDE_RETREAT,
        )
        self._dbg("B placed cup")
        self.delay(30, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        p = self.cup.get_pose()
        target_xy = np.array(PLACE_TARGET.p[:2], dtype=float)
        cup_xy = np.array(p.p[:2], dtype=float)
        target_err = float(np.linalg.norm(cup_xy - target_xy))
        height_err = abs(float(p.p[2]) - float(PLACE_TARGET.p[2]))
        up = float(np.dot(p.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.75

        self.metadata["cup"] = [float(v) for v in p.p]
        self.metadata["target_err"] = target_err
        self.metadata["height_err"] = height_err
        print(
            f"[HANDOVER_PLACE] target_err={target_err*1000:.1f}mm "
            f"height_err={height_err*1000:.1f}mm up={up}",
            flush=True,
        )
        return bool(target_err < 0.035 and height_err < 0.025 and up)
