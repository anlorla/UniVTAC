from ._base_task import *
import numpy as np

# ============================================================================
# Dual Plate PLACE-STACK —— 双臂叠放浅盘:
#   初始: 两个盘子分别放在左右两侧, 盘面朝 local +Z。
#   流程: A 抓 plate_a 的 +Y 侧盘沿并放到中间目标位; B 抓 plate_b 的 -Y 侧盘沿
#         并放到 plate_a 正上方形成叠放。
# ----------------------------------------------------------------------------
#   对象文件:
#     plate_a/plate_b : assets/objects/PLATE.usd
#       - 仓库未保留任务用 PLATE.obj; 可由 scripts/asset_tools/build_plate.py
#         重新生成到 /tmp/plate_assets/PLATE.obj。
#       - PLATE.usd 内可见 tet_* 属性, 用于 UIPC 可变形/接触仿真。
#       - 若未来需要重建资产: build_plate.py -> bake_tet.py -> write_tet_to_usd.py。
#   对象几何:
#     PLATE.usd 是浅圆盘/托盘, 外径约 200mm, 边沿高约 8mm, 原点在盘体中心,
#     local z 范围约 [-4mm, +4mm]。
#   触觉说明:
#     只用 USD 也可以采触觉, 前提是 USD 里有可供 UIPC 加载的网格/tet 信息。
#     本任务通过 add_from_usd_file("PLATE.usd") 创建 UIPC actor, GelSight 与盘沿接触
#     时仍会记录 tactile rgb/depth/marker/pose。
# ============================================================================

# ---- 几何 (m) ----
PLATE_HALF_Z = 0.004
PLATE_R = 0.100
RIM_GRASP_R = 0.098
RIM_GRASP_DZ = PLATE_HALF_Z - 0.002
TABLE_TOP = 0.004
GAP = 0.006
STACK_DZ = 0.010

# ---- 摆位 (世界系) ----
UPRIGHT = [1, 0, 0, 0]
PLATE_A_START = Pose([0.42, 0.26, TABLE_TOP + PLATE_HALF_Z + GAP], UPRIGHT)
PLATE_B_START = Pose([0.42, -0.26, TABLE_TOP + PLATE_HALF_Z + GAP], UPRIGHT)
STACK_TARGET = Pose([0.50, 0.00, TABLE_TOP + PLATE_HALF_Z], UPRIGHT)
STACK_TOP_TARGET = Pose(
    [STACK_TARGET.p[0], STACK_TARGET.p[1], STACK_TARGET.p[2] + STACK_DZ],
    UPRIGHT,
)

# ---- 动作参数 ----
GRASP_PRE_DIS = 0.12
PLACE_HOVER = 0.10
RETRACT_Z = 0.10
ARM_A_PARK_POS = [0.36, 0.31, 0.24]


@configclass
class TaskCfg(BaseTaskCfg):
    dual_arm = True
    use_adaptive_grasp = False
    viewer = ViewerCfg(eye=(1.45, 0.0, 0.95), lookat=(0.50, 0.0, 0.08))
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
        # 保留 plate_b 作为初始化静态障碍, 避免 cuRobo 空障碍分支; 两臂运行期忽略被操作盘。
        self.planner_ignore_actors = {"plate_a"}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {"plate_a", "plate_b"}
        self._robot_manager_b.ignore_actors = {"plate_a", "plate_b"}

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        density = 2e5
        self.plate_a = self._actor_manager.add_from_usd_file(
            name="plate_a",
            asset_path="PLATE.usd",
            pose=PLATE_A_START,
            density=density,
        )
        self.plate_b = self._actor_manager.add_from_usd_file(
            name="plate_b",
            asset_path="PLATE.usd",
            pose=PLATE_B_START,
            density=density,
        )

    def _reset_actors(self):
        noise_a = self.create_noise([0.003, 0.003, 0.0])
        noise_b = self.create_noise([0.003, 0.003, 0.0])
        self.plate_a.set_pose(PLATE_A_START.add_offset(noise_a))
        self.plate_b.set_pose(PLATE_B_START.add_offset(noise_b))
        self.metadata["target_pose"] = [float(v) for v in STACK_TARGET.p]

    # ---------------------------------------------------------------- helpers
    def _register_rim_grasp(self, actor, rim_dir):
        rd = np.array(rim_dir, dtype=float)
        rd /= np.linalg.norm(rd)
        center = actor.get_pose()
        gp = np.array(center.p, dtype=float) + rd * RIM_GRASP_R + np.array([0.0, 0.0, RIM_GRASP_DZ])
        grasp_pose = construct_grasp_pose(
            gp,
            np.array([0.0, 0.0, 1.0], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
        )
        return actor.register_point(grasp_pose, type="contact")

    def _grasp_plate(self, actor, atom, rm, arm, rim_dir):
        self.move(atom.open_gripper(1.0), arm=arm)
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

    def _release_plate(self, actor, atom, arm, park=False):
        self._unweld_actor(actor)
        self.delay(5, is_save=True)
        self.move(atom.open_gripper(1.0), arm=arm)
        self.move(
            atom.move_by_displacement(z=RETRACT_Z, xyz_coord="world"),
            arm=arm,
            constraint_pose=[1, 1, 1, 1, 1, 0],
            time_dilation_factor=0.5,
        )
        if park:
            rm, _ = self._arm(arm)
            gc_now = rm.get_gripper_center_pose()
            park_gc = Pose(ARM_A_PARK_POS, gc_now.q)
            self.move(
                atom.move_to_pose(rm.gripper_center_to_ee(park_gc)),
                arm=arm,
                time_dilation_factor=0.5,
            )

    def _dbg(self, tag):
        ap, bp = self.plate_a.get_pose(), self.plate_b.get_pose()
        print(
            f"[PLATE_STACK] {tag}: "
            f"plate_a=({ap.p[0]:.3f},{ap.p[1]:.3f},{ap.p[2]:.3f}) "
            f"plate_b=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f})",
            flush=True,
        )

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(15)
        self._dbg("reset settled")

    def _play_once(self):
        # A: 把下盘从 +Y 侧搬到中间目标位。
        self._grasp_plate(self.plate_a, self.atom_a, self._robot_manager, "a", rim_dir=(0, 1, 0))
        self._place_inhand(
            self.plate_a,
            self._robot_manager,
            self.atom_a,
            STACK_TARGET.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "a",
        )
        self._place_inhand(self.plate_a, self._robot_manager, self.atom_a, STACK_TARGET, "a")
        self._release_plate(self.plate_a, self.atom_a, "a", park=True)
        self._dbg("A placed plate_a")

        # B: 把上盘从 -Y 侧搬到 plate_a 正上方。
        self._grasp_plate(self.plate_b, self.atom_b, self._robot_manager_b, "b", rim_dir=(0, -1, 0))
        self._place_inhand(
            self.plate_b,
            self._robot_manager_b,
            self.atom_b,
            STACK_TOP_TARGET.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "b",
        )
        self._place_inhand(self.plate_b, self._robot_manager_b, self.atom_b, STACK_TOP_TARGET, "b")
        self._release_plate(self.plate_b, self.atom_b, "b")
        self._dbg("B stacked plate_b")
        self.delay(30, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        ap = self.plate_a.get_pose()
        bp = self.plate_b.get_pose()
        target_xy = np.array(STACK_TARGET.p[:2], dtype=float)
        a_xy = np.array(ap.p[:2], dtype=float)
        b_xy = np.array(bp.p[:2], dtype=float)
        target_err = float(np.linalg.norm(a_xy - target_xy))
        stack_err = float(np.linalg.norm(b_xy - a_xy))
        dz = float(bp.p[2] - ap.p[2])
        height_err = abs(dz - STACK_DZ)
        a_up = float(np.dot(ap.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.8
        b_up = float(np.dot(bp.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.8

        self.metadata["plate_a"] = [float(v) for v in ap.p]
        self.metadata["plate_b"] = [float(v) for v in bp.p]
        self.metadata["target_err"] = target_err
        self.metadata["stack_err"] = stack_err
        self.metadata["height_err"] = height_err
        print(
            f"[PLATE_STACK] target_err={target_err*1000:.1f}mm "
            f"stack_err={stack_err*1000:.1f}mm dz={dz*1000:.1f}mm "
            f"height_err={height_err*1000:.1f}mm a_up={a_up} b_up={b_up}",
            flush=True,
        )
        return bool(target_err < 0.04 and stack_err < 0.03 and height_err < 0.015 and a_up and b_up)
