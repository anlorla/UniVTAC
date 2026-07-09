from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Dual Cup PLACE-STACK —— 双臂叠放纸杯:
#   初始: 两个杯子分别放在左右两侧, 杯口朝 local +Z, 闭底朝 -Z。
#   流程: A 抓 cup_a 并放到中间目标位; B 抓 cup_b 并放到 cup_a 正上方形成套叠。
#     cup_a/cup_b : assets/objects/CUP.usd
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
OVERLAP = 0.040 * CUP_SCALE
NEST_RISE = 2 * CUP_HALF - OVERLAP
TABLE_TOP = 0.004
GAP = 0.006
STACK_GAP = 0.0

# ---- 摆位 (世界系) ----
UPRIGHT = [1, 0, 0, 0]
CUP_A_START = Pose([0.40, 0.22, TABLE_TOP + CUP_HALF + GAP], UPRIGHT)
CUP_B_START = Pose([0.40, -0.22, TABLE_TOP + CUP_HALF + GAP], UPRIGHT)
STACK_TARGET = Pose([0.48, 0.00, TABLE_TOP + CUP_HALF + GAP], UPRIGHT)
STACK_TOP_TARGET = Pose(
    [STACK_TARGET.p[0], STACK_TARGET.p[1], STACK_TARGET.p[2] + NEST_RISE + STACK_GAP],
    UPRIGHT,
)

# ---- 动作参数 ----
GRASP_R = CUP_TOP_R + 0.002
GRASP_DZ = CUP_HALF - 0.011
GRASP_PRE_DIS = 0.12
PRE_GRIPPER_OPEN = 0.62
PLACE_HOVER = 0.11
RETRACT_Z = 0.10
ARM_A_PARK_POS = [0.36, 0.30, 0.24]
GRIPPER_RELEASE_THRESH = 0.65
SUCCESS_HOLD_STEPS = 8


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
    step_lim = 1400
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
        # cup_a 是 A 的在手/目标件, 初始化规划世界时必须忽略; cup_b 暂留作静态障碍,
        # 避免 cuRobo 在只有地面时触发 "Primitive Collision has no obstacles" 分支。
        # 后续 B 臂的运行期 ignore 集仍会包含 cup_a/cup_b。
        self.planner_ignore_actors = {"cup_a"}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {"cup_a", "cup_b"}
        self._robot_manager_b.ignore_actors = {"cup_a", "cup_b"}

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        sc = (CUP_SCALE, CUP_SCALE, CUP_SCALE)
        cup_density = 2e5
        self.cup_a = self._actor_manager.add_from_usd_file(
            name="cup_a",
            asset_path="CUP.usd",
            pose=CUP_A_START,
            scale=sc,
            density=cup_density,
        )
        self.cup_b = self._actor_manager.add_from_usd_file(
            name="cup_b",
            asset_path="CUP.usd",
            pose=CUP_B_START,
            scale=sc,
            density=cup_density,
        )

    def _reset_actors(self):
        noise_a = self.create_noise([0.003, 0.003, 0.0])
        noise_b = self.create_noise([0.003, 0.003, 0.0])
        self.cup_a.set_pose(CUP_A_START.add_offset(noise_a))
        self.cup_b.set_pose(CUP_B_START.add_offset(noise_b))
        self._success_hold_count = 0
        self.metadata["target_pose"] = [float(v) for v in STACK_TARGET.p]

    # ---------------------------------------------------------------- helpers
    def _register_rim_grasp(self, actor, rim_dir):
        rd = np.array(rim_dir, dtype=float)
        rd /= np.linalg.norm(rd)
        center = actor.get_pose()
        gp = np.array(center.p, dtype=float) + rd * GRASP_R + np.array([0.0, 0.0, GRASP_DZ])
        grasp_pose = construct_grasp_pose(
            gp,
            np.array([0.0, 0.0, 1.0], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
        )
        return actor.register_point(grasp_pose, type="contact")

    def _grasp_cup(self, actor, atom, rm, arm, rim_dir):
        self.move(atom.open_gripper(PRE_GRIPPER_OPEN), arm=arm)
        grasp_id = self._register_rim_grasp(actor, rim_dir)
        self.move(
            atom.grasp_actor(
                actor,
                contact_point_id=grasp_id,
                pre_dis=GRASP_PRE_DIS,
                dis=0.0,
                is_close=False,
            ),
            arm=arm,
            time_dilation_factor=0.5,
        )
        self.move(atom.close_gripper(0.0, depth_threshold=None), arm=arm)
        self.weld_actor(actor, rm)
        self.delay(8, is_save=True)

    def _place_inhand(self, actor, rm, atom, target_pose, arm):
        inhand = actor.get_pose().rebase(rm.get_gripper_center_pose())
        gc_mat = np.array(target_pose.to_transformation_matrix()) @ np.linalg.inv(
            np.array(inhand.to_transformation_matrix())
        )
        ee = rm.gripper_center_to_ee(Pose.from_matrix(gc_mat))
        self.move(atom.move_to_pose(ee), arm=arm, time_dilation_factor=0.5)

    def _unweld_actor(self, actor):
        self._welds = [w for w in self._welds if w[0] is not actor]

    def _cups_released(self):
        a_open = float(self._robot_manager.get_gripper_percentage()) >= GRIPPER_RELEASE_THRESH
        b_open = float(self._robot_manager_b.get_gripper_percentage()) >= GRIPPER_RELEASE_THRESH
        no_cup_welds = not any(w[0] is self.cup_a or w[0] is self.cup_b for w in self._welds)
        return a_open, b_open, no_cup_welds, bool(a_open and b_open and no_cup_welds)

    def _release_cup(self, actor, atom, arm, park=False, retract=True):
        self._unweld_actor(actor)
        self.delay(5, is_save=True)
        self.move(atom.open_gripper(1.0), arm=arm)
        if retract:
            self.move(
                atom.move_by_displacement(z=RETRACT_Z, xyz_coord="world"),
                arm=arm,
                time_dilation_factor=0.5,
            )
        if park:
            # 双臂叠放时 A 臂若停在中间目标上方, B 臂过来放第二个杯子会和 A 反复碰撞。
            # 释放并上抬后停到左侧上方的明确停车位; 不用 back_to_origin(), 避免视觉上退得过远。
            rm, _ = self._arm(arm)
            gc_now = rm.get_gripper_center_pose()
            park_gc = Pose(ARM_A_PARK_POS, gc_now.q)
            self.move(
                atom.move_to_pose(rm.gripper_center_to_ee(park_gc)),
                arm=arm,
                time_dilation_factor=0.5,
            )

    def _dbg(self, tag):
        ap, bp = self.cup_a.get_pose(), self.cup_b.get_pose()
        print(
            f"[PLACE_STACK] {tag}: "
            f"cup_a=({ap.p[0]:.3f},{ap.p[1]:.3f},{ap.p[2]:.3f}) "
            f"cup_b=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f})",
            flush=True,
        )

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(15)
        self._dbg("reset settled")

    def _play_once(self):
        # A: 把下杯从 +Y 侧搬到中间目标位。
        self._grasp_cup(self.cup_a, self.atom_a, self._robot_manager, "a", rim_dir=(0, 1, 0))
        self._place_inhand(
            self.cup_a,
            self._robot_manager,
            self.atom_a,
            STACK_TARGET.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "a",
        )
        self._place_inhand(self.cup_a, self._robot_manager, self.atom_a, STACK_TARGET, "a")
        self._release_cup(self.cup_a, self.atom_a, "a", park=True)
        self._dbg("A placed cup_a")

        # B: 把上杯从 -Y 侧搬到 cup_a 正上方, 形成套叠。
        self._grasp_cup(self.cup_b, self.atom_b, self._robot_manager_b, "b", rim_dir=(0, -1, 0))
        self._place_inhand(
            self.cup_b,
            self._robot_manager_b,
            self.atom_b,
            STACK_TOP_TARGET.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "b",
        )
        self._place_inhand(self.cup_b, self._robot_manager_b, self.atom_b, STACK_TOP_TARGET, "b")
        self._release_cup(self.cup_b, self.atom_b, "b", retract=False)
        self._dbg("B stacked cup_b")
        self.delay(30, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        ap = self.cup_a.get_pose()
        bp = self.cup_b.get_pose()
        target_xy = np.array(STACK_TARGET.p[:2], dtype=float)
        a_xy = np.array(ap.p[:2], dtype=float)
        b_xy = np.array(bp.p[:2], dtype=float)
        target_err = float(np.linalg.norm(a_xy - target_xy))
        stack_err = float(np.linalg.norm(b_xy - a_xy))
        dz = float(bp.p[2] - ap.p[2])
        height_err = abs(dz - NEST_RISE)
        a_up = float(np.dot(ap.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.75
        b_up = float(np.dot(bp.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.75
        a_open, b_open, no_cup_welds, released = self._cups_released()
        geom_ok = target_err < 0.035 and stack_err < 0.025 and height_err < 0.025 and a_up and b_up
        if geom_ok and released:
            self._success_hold_count = getattr(self, "_success_hold_count", 0) + 1
        else:
            self._success_hold_count = 0

        self.metadata["cup_a"] = [float(v) for v in ap.p]
        self.metadata["cup_b"] = [float(v) for v in bp.p]
        self.metadata["target_err"] = target_err
        self.metadata["stack_err"] = stack_err
        self.metadata["height_err"] = height_err
        self.metadata["gripper_a_open"] = a_open
        self.metadata["gripper_b_open"] = b_open
        self.metadata["cups_unwelded"] = no_cup_welds
        self.metadata["released"] = released
        self.metadata["success_hold_count"] = int(self._success_hold_count)
        print(
            f"[PLACE_STACK] target_err={target_err*1000:.1f}mm "
            f"stack_err={stack_err*1000:.1f}mm dz={dz*1000:.1f}mm "
            f"height_err={height_err*1000:.1f}mm a_up={a_up} b_up={b_up} "
            f"a_open={a_open:.2f} b_open={b_open:.2f} unwelded={no_cup_welds} "
            f"hold={self._success_hold_count}/{SUCCESS_HOLD_STEPS}",
            flush=True,
        )
        return bool(self._success_hold_count >= SUCCESS_HOLD_STEPS)
