from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Dual Bowl PLACE-STACK —— 双臂叠放陶碗 (仿 dual_cup_place_stack, 物体换成碗):
#   初始: 两只碗分别放在左右两侧桌面, 碗口朝 local +Z, 闭底朝 -Z。
#   流程: A 抓 bowl_a 搬到中间目标位放下; B 抓 bowl_b 放到 bowl_a 正上方形成套叠。
#     bowl_a/bowl_b : assets/objects/BOWL.usd (⌀130mm 浅口陶碗, 5mm 壁, 带圈足;
#       碗口朝 local +Z, 闭底 -Z, 体心居中; USD 已是实尺 -> scale=1.0)
#   与纸杯版的关键区别 (同 dual_bowl_unstack):
#     碗口 ⌀130mm > 夹爪最大开度, 无法整只跨抱 -> 必须只夹住【碗沿】一侧
#     (一指落碗内壁、一指碗外壁, 沿径向钳夹口下约 5mm 处约 5mm 厚的碗壁)。
#     不用自适应抓取: use_adaptive_grasp=False + close_gripper(0.0) 直接 100% 闭合,
#     抓住后 weld_actor 把碗刚性绑到夹爪上, 搬运全程不滑脱。
# ----------------------------------------------------------------------------
#   BOWL 局部(原点居中, z∈[-half,+half]): 碗口 local +Z, 闭底 local -Z; 原点->口沿 = BOWL_HALF。
# ============================================================================

# ---- 几何 (m) ----
BOWL_HALF    = 0.0197    # 碗半高(原点->口沿 / 原点->底); USD 已是 ⌀130mm/高39.4mm 实尺, scale=1.0
BOWL_OUTER_R = 0.065     # 碗口外半径(⌀130mm)
RIM_GRASP_R  = 0.060     # 夹爪中心(TCP)到碗轴的水平距离(落在口下约 5mm 处壁厚中线)
FOOT_R       = 0.026     # 圈足外半径
TABLE_TOP    = 0.004     # 桌面顶高(碗底贴此面); 碗体心 = TABLE_TOP + BOWL_HALF
GAP          = 0.006     # 初始体心离桌面的小间隙(放置/抓取留余量)
# 套叠抬高量(同 dual_bowl_unstack): 上碗坐进下碗后体心比下碗高 NEST_RISE。
NEST_RISE    = 0.038
STACK_GAP    = 0.0

# ---- 摆位 (世界系) ----
UPRIGHT      = [1, 0, 0, 0]
BOWL_A_START = Pose([0.42,  0.26, TABLE_TOP + BOWL_HALF + GAP], UPRIGHT)   # 下碗在 +Y 侧(A 臂)
BOWL_B_START = Pose([0.42, -0.26, TABLE_TOP + BOWL_HALF + GAP], UPRIGHT)   # 上碗在 -Y 侧(B 臂)
STACK_TARGET = Pose([0.48,  0.00, TABLE_TOP + BOWL_HALF + GAP], UPRIGHT)   # 下碗目标位(中间)
STACK_TOP_TARGET = Pose(
    [STACK_TARGET.p[0], STACK_TARGET.p[1], STACK_TARGET.p[2] + NEST_RISE + STACK_GAP],
    UPRIGHT,
)

# ---- 动作参数 ----
RIM_GRASP_DZ    = BOWL_HALF - 0.006   # 抓取点竖直偏置(口沿顶下方约 6mm, 落在实壁上)
GRASP_PRE_DIS   = 0.12                # 抓取前沿接近轴的悬停距离
PRE_GRIPPER_OPEN = 1.0                # 碗沿抓取: 先全开, 让两指能跨在内外壁两侧
PLACE_HOVER     = 0.12                # 放置目标正上方的悬停高度
RETRACT_Z       = 0.12                # 松爪后竖直撤离高度
ARM_A_PARK_POS  = [0.40, 0.32, 0.26]  # A 放完下碗后退到 +Y 上方的停车位, 让开 B 臂


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
    # 双臂 + 碗沿抓取(分段受约束直线进刀) + 套叠落点, 比纸杯版略长;
    # 两个抓-放周期约 1500 步, 把 step_lim / max_save_frames 都给足防中途截断。
    step_lim = 1600
    max_save_frames = 1600
    reset_time_limit = 1200.0


class Task(BaseTask):
    def __init__(
        self,
        cfg: BaseTaskCfg,
        mode: Literal["collect", "eval"] = "collect",
        render_mode: str | None = None,
        **kwargs,
    ):
        # 中高摩擦: 利于夹住碗沿(线接触), 同时不像超高摩擦那样加重接触求解负担。
        cfg.sim.physics_material.dynamic_friction = 5.0
        cfg.sim.physics_material.static_friction = 5.0
        cfg.uipc_sim.contact.default_friction_ratio = 5.0
        # bowl_a 是 A 的在手/目标件, 初始化规划世界时必须忽略; bowl_b 暂留作静态障碍,
        # 避免 cuRobo 在只有地面时触发 "Primitive Collision has no obstacles" 分支。
        self.planner_ignore_actors = {"bowl_a"}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {"bowl_a", "bowl_b"}
        self._robot_manager_b.ignore_actors = {"bowl_a", "bowl_b"}

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        # 下碗(bowl_a)放完后要当底座扛住上碗套入 -> 高密度+高摩擦留得住; 上碗(bowl_b)轻便于操作。
        self.bowl_a = self._actor_manager.add_from_usd_file(
            name="bowl_a",
            asset_path="BOWL.usd",
            pose=BOWL_A_START,
            density=2e5,
        )
        self.bowl_b = self._actor_manager.add_from_usd_file(
            name="bowl_b",
            asset_path="BOWL.usd",
            pose=BOWL_B_START,
            density=5e3,
        )

    def _reset_actors(self):
        noise_a = self.create_noise([0.003, 0.003, 0.0])
        noise_b = self.create_noise([0.003, 0.003, 0.0])
        self.bowl_a.set_pose(BOWL_A_START.add_offset(noise_a))
        self.bowl_b.set_pose(BOWL_B_START.add_offset(noise_b))
        self.metadata["target_pose"] = [float(v) for v in STACK_TARGET.p]

    # ---------------------------------------------------------------- helpers
    def _register_rim_grasp(self, actor, rim_dir):
        # 抓取点 = 碗体心 + 径向 RIM_GRASP_R(口下约 5mm 处壁厚中线) + 竖直 RIM_GRASP_DZ。
        # camera_up=(1,0,0) 使夹爪开合方向 = 径向(rim_dir): 一指落碗内壁、一指碗外壁。
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

    def _grasp_bowl(self, actor, atom, rm, arm, rim_dir):
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
        # 碗沿无法自适应判深 -> 直接 100% 闭合到底, 再 weld 把碗刚性绑到夹爪(搬运不滑脱)。
        self.move(atom.close_gripper(0.0, depth_threshold=None), arm=arm)
        self.weld_actor(actor, rm)
        self.delay(8, is_save=True)

    def _place_inhand(self, actor, rm, atom, target_pose, arm):
        # 用当前在手相对位姿反解夹爪中心目标 -> 即使碗沿离心抓持, 也能把碗体心摆到 target。
        inhand = actor.get_pose().rebase(rm.get_gripper_center_pose())
        gc_mat = np.array(target_pose.to_transformation_matrix()) @ np.linalg.inv(
            np.array(inhand.to_transformation_matrix())
        )
        ee = rm.gripper_center_to_ee(Pose.from_matrix(gc_mat))
        self.move(atom.move_to_pose(ee), arm=arm, time_dilation_factor=0.5)

    def _unweld_actor(self, actor):
        self._welds = [w for w in self._welds if w[0] is not actor]

    def _release_bowl(self, actor, atom, arm, park=False, retract=True):
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
            # A 臂放完下碗后若停在中间目标上方, B 臂过来套上碗会反复碰撞 -> 退到 +Y 上方停车位。
            rm, _ = self._arm(arm)
            gc_now = rm.get_gripper_center_pose()
            park_gc = Pose(ARM_A_PARK_POS, gc_now.q)
            self.move(
                atom.move_to_pose(rm.gripper_center_to_ee(park_gc)),
                arm=arm,
                time_dilation_factor=0.5,
            )

    def _dbg(self, tag):
        ap, bp = self.bowl_a.get_pose(), self.bowl_b.get_pose()
        print(
            f"[BOWL_STACK] {tag}: "
            f"bowl_a=({ap.p[0]:.3f},{ap.p[1]:.3f},{ap.p[2]:.3f}) "
            f"bowl_b=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f})",
            flush=True,
        )

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(15)
        self._dbg("reset settled")

    def _play_once(self):
        # A: 把下碗从 +Y 侧搬到中间目标位。
        self._grasp_bowl(self.bowl_a, self.atom_a, self._robot_manager, "a", rim_dir=(0, 1, 0))
        self._place_inhand(
            self.bowl_a,
            self._robot_manager,
            self.atom_a,
            STACK_TARGET.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "a",
        )
        self._place_inhand(self.bowl_a, self._robot_manager, self.atom_a, STACK_TARGET, "a")
        self._release_bowl(self.bowl_a, self.atom_a, "a", park=True)
        self._dbg("A placed bowl_a")

        # B: 把上碗从 -Y 侧搬到 bowl_a 正上方, 套入形成套叠。
        self._grasp_bowl(self.bowl_b, self.atom_b, self._robot_manager_b, "b", rim_dir=(0, -1, 0))
        self._place_inhand(
            self.bowl_b,
            self._robot_manager_b,
            self.atom_b,
            STACK_TOP_TARGET.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "b",
        )
        self._place_inhand(self.bowl_b, self._robot_manager_b, self.atom_b, STACK_TOP_TARGET, "b")
        self._release_bowl(self.bowl_b, self.atom_b, "b", retract=False)
        self._dbg("B stacked bowl_b")
        self.delay(30, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        ap = self.bowl_a.get_pose()
        bp = self.bowl_b.get_pose()
        target_xy = np.array(STACK_TARGET.p[:2], dtype=float)
        a_xy = np.array(ap.p[:2], dtype=float)
        b_xy = np.array(bp.p[:2], dtype=float)
        target_err = float(np.linalg.norm(a_xy - target_xy))
        stack_err = float(np.linalg.norm(b_xy - a_xy))
        dz = float(bp.p[2] - ap.p[2])
        height_err = abs(dz - NEST_RISE)
        a_up = float(np.dot(ap.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.75
        b_up = float(np.dot(bp.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.75

        self.metadata["bowl_a"] = [float(v) for v in ap.p]
        self.metadata["bowl_b"] = [float(v) for v in bp.p]
        self.metadata["target_err"] = target_err
        self.metadata["stack_err"] = stack_err
        self.metadata["height_err"] = height_err
        print(
            f"[BOWL_STACK] target_err={target_err*1000:.1f}mm "
            f"stack_err={stack_err*1000:.1f}mm dz={dz*1000:.1f}mm "
            f"height_err={height_err*1000:.1f}mm a_up={a_up} b_up={b_up}",
            flush=True,
        )
        # 碗比纸杯宽, 落点/套叠容差稍放宽。
        return bool(target_err < 0.04 and stack_err < 0.03 and height_err < 0.025 and a_up and b_up)
