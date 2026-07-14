from ._base_task import *
import numpy as np
import torch

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

# 去 weld 开关: 盘改成【更大更矮壁的碗】(竖直外壁)后, 夹爪闭到底被 4.5mm 壁挡住形成强力钳夹,
# 靠接触摩擦即可扛住搬运 -> 不再 weld。改 True 可回退到旧的刚性绑定。
USE_WELD = False

# ---- 几何 (m) ---- (PLATE_STACK.usd = Ø190mm/高40mm 竖壁浅碗, 实尺 scale=1.0)
PLATE_HALF_Z = 0.020     # 盘半高(原点->沿口 / 原点->底)
PLATE_R = 0.095          # 盘口外半径(Ø190mm)
RIM_GRASP_R = 0.091      # 夹爪中心(TCP)到盘轴的水平距离(竖直壁厚中线 r≈91mm, 8mm 壁 87~95)
RIM_GRASP_DZ = PLATE_HALF_Z - 0.014   # 抓取点竖直偏置(沿口下方约 14mm, 落在竖直壁带上)
RIM_DOWN_EXTRA = 0.014   # 到位后再多压 14mm 让胶垫贴实(40mm 壁给足余量, 又不顶桌面)
GRASP_SIDE = 0.05        # 悬停高度分量(side)
GRASP_UP = 0.06          # 悬停高度分量(up)
GRASP_LIFT = 0.10        # 抓稳后先竖直抬起的高度(再横移)
TABLE_TOP = 0.004
GAP = 0.006
STACK_DZ = 0.028         # 套叠抬高量(H40 竖壁; 见 check_success 用范围判)

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
SUCCESS_HOLD_STEPS = 8                 # 需连续 N 步维持"叠放正立 + 双爪释放"才计成功(仿 cup 版)
GRIPPER_RELEASE_THRESH = 0.65          # 夹爪开合百分比 >= 此值视为已松开


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
    # 去 weld 后每次抓取多了压实+底层强制夹紧+settle, 两周期约 1700 步 -> 给足 2600 防截断。
    step_lim = 2600
    max_save_frames = 2600
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
        # 去 weld 后薄壁钳夹要扛住整只盘: 下盘大幅减重(2e5 会把夹持拉歪/滑脱), 只需比上盘略重当底座。
        self.plate_a = self._actor_manager.add_from_usd_file(
            name="plate_a",
            asset_path="PLATE_STACK.usd",
            pose=PLATE_A_START,
            density=1e4,
        )
        self.plate_b = self._actor_manager.add_from_usd_file(
            name="plate_b",
            asset_path="PLATE_STACK.usd",
            pose=PLATE_B_START,
            density=5e3,
        )

    def _reset_actors(self):
        noise_a = self.create_noise([0.003, 0.003, 0.0])
        noise_b = self.create_noise([0.003, 0.003, 0.0])
        self.plate_a.set_pose(PLATE_A_START.add_offset(noise_a))
        self.plate_b.set_pose(PLATE_B_START.add_offset(noise_b))
        self.metadata["target_pose"] = [float(v) for v in STACK_TARGET.p]
        self._success_hold_count = 0

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

    def _close_gripper_direct(self, rm, percent=0.0, settle_steps=10, is_save=True):
        # 底层强制闭合并逐步 settle: 让两指切实压紧薄盘壁(仅 close_gripper 规划闭合往往夹不实)。
        target = rm.gripper_percent2qpos(percent)
        pos = torch.tensor([target, target], device=rm.device, dtype=rm.robot.data.joint_pos.dtype)
        self.atom_id += 1
        self.atom_tag = "close_direct"
        rm.set_gripper(pos, force=True)
        for _ in range(settle_steps):
            rm.set_gripper(pos, force=True)
            self._step(is_save=is_save)
        self._update_render()

    def _open_gripper_direct(self, rm, percent=1.0, settle_steps=12, is_save=True):
        target = rm.gripper_percent2qpos(percent)
        pos = torch.tensor([target, target], device=rm.device, dtype=rm.robot.data.joint_pos.dtype)
        self.atom_id += 1
        self.atom_tag = "open_direct"
        rm.set_gripper(pos, force=True)
        for _ in range(settle_steps):
            rm.set_gripper(pos, force=True)
            self._step(is_save=is_save)
        self._update_render()

    def _grasp_plate(self, actor, atom, rm, arm, rim_dir, camera_up=(1, 0, 0)):
        # 照抄 dual_bowl_place_stack/unstack 的成功配方(no-weld 也抓得住):
        # 悬停 -> 受约束直线下探(多压 RIM_DOWN_EXTRA 让胶垫贴实内外壁) -> 规划闭合 + 底层强制夹紧 -> 先抬后移。
        # camera_up 决定腕部朝向: A(+Y臂)用 [1,0,0]; 镜像的 B(-Y臂)在宽盘抓取点上需 [-1,0,0] 才可达
        # (手指仍沿 Y 夹壁, 只是腕翻到 B 的自然侧)。
        rd = np.array(rim_dir, dtype=float); rd = rd / np.linalg.norm(rd)
        self.move(atom.open_gripper(1.0), arm=arm)
        center = actor.get_pose()
        gp = np.array(center.p, dtype=float) + rd * RIM_GRASP_R + np.array([0.0, 0.0, RIM_GRASP_DZ])
        print(f"[PLATE_STACK] {arm} grasp target gp=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
              f"center=({center.p[0]:.3f},{center.p[1]:.3f}) rim_dir={tuple(rim_dir)} cam_up={tuple(camera_up)}", flush=True)
        gc = construct_grasp_pose(gp, np.array([0, 0, 1], dtype=float), np.array(camera_up, dtype=float))
        gc_high = gc.add_bias([0.0, 0.0, GRASP_SIDE + GRASP_UP], coord="world")
        self.move(atom.move_to_pose(rm.gripper_center_to_ee(gc_high)), arm=arm)
        down_total = GRASP_SIDE + GRASP_UP + RIM_DOWN_EXTRA
        self.move(atom.move_by_displacement(z=down_total * 0.75, xyz_coord="local"), arm=arm,
                  constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(atom.move_by_displacement(z=down_total * 0.25, xyz_coord="local"), arm=arm,
                  constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(atom.close_gripper(0.0, depth_threshold=None), arm=arm)
        self._close_gripper_direct(rm, 0.0, settle_steps=10, is_save=True)
        if USE_WELD:
            self.weld_actor(actor, rm)
        self.delay(8, is_save=True)
        self.move(
            atom.move_by_displacement(z=GRASP_LIFT, xyz_coord="world"),
            arm=arm,
            time_dilation_factor=0.5,
        )
        gc_now = rm.get_gripper_center_pose()
        bp = actor.get_pose()
        print(
            f"[PLATE_STACK] {arm} grasp+lift: plate=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f}) "
            f"gc=({gc_now.p[0]:.3f},{gc_now.p[1]:.3f},{gc_now.p[2]:.3f}) "
            f"held={'YES' if bp.p[2] > 0.08 else 'NO(dropped)'}",
            flush=True,
        )

    def _place_inhand(self, actor, rm, atom, target_pose, arm):
        inhand = actor.get_pose().rebase(rm.get_gripper_center_pose())
        gc_mat = np.array(target_pose.to_transformation_matrix()) @ np.linalg.inv(
            np.array(inhand.to_transformation_matrix())
        )
        ee = rm.gripper_center_to_ee(Pose.from_matrix(gc_mat))
        self.move(atom.move_to_pose(ee), arm=arm, time_dilation_factor=0.5)

    def _unweld_actor(self, actor):
        self._welds = [w for w in self._welds if w[0] is not actor]

    def _plates_released(self):
        # 两盘都真正脱手 = 两爪都张开 且 两盘都已 unweld(搬运焊接解除)。
        a_open = float(self._robot_manager.get_gripper_percentage()) >= GRIPPER_RELEASE_THRESH
        b_open = float(self._robot_manager_b.get_gripper_percentage()) >= GRIPPER_RELEASE_THRESH
        no_plate_welds = not any(w[0] is self.plate_a or w[0] is self.plate_b for w in self._welds)
        return a_open, b_open, no_plate_welds, bool(a_open and b_open and no_plate_welds)

    def _release_plate(self, actor, atom, arm, park=False):
        rm, _ = self._arm(arm)
        self._unweld_actor(actor)
        self.delay(5, is_save=True)
        self._open_gripper_direct(rm, 1.0, settle_steps=12, is_save=True)
        # 注意: 竖直回撤【不要】加 constraint_pose=[1,1,1,1,1,0] —— 锁 5 轴的约束回撤在放盘后的
        # 位形常规划不出而报错, 而框架里任何一步 move 失败会【跳过后续所有 move】(于是 B 整段被跳过、
        # 永远抓不到)。改成普通世界系竖直位移(照抄 bowl 的 _release_bowl)。
        self.move(
            atom.move_by_displacement(z=RETRACT_Z, xyz_coord="world"),
            arm=arm,
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

        # B: 把上盘从 -Y 侧搬到 plate_a 正上方。抓最远 -Y 沿(A 的镜像对称, 同 bowl 的成功配置)。
        self._grasp_plate(self.plate_b, self.atom_b, self._robot_manager_b, "b",
                          rim_dir=(0, -1, 0), camera_up=(1, 0, 0))
        # 关键: 对准 plate_a 的【实际落点】(而非名义 STACK_TARGET)。plate_a 带噪声/微偏, 若 B 固定往
        # 名义中心放 -> 盘沿错位相撞把底盘撞歪(同叠杯的失败机理)。
        pa = self.plate_a.get_pose()
        top_target = Pose(
            [float(pa.p[0]), float(pa.p[1]), float(pa.p[2]) + STACK_DZ],
            UPRIGHT,
        )
        self._place_inhand(
            self.plate_b,
            self._robot_manager_b,
            self.atom_b,
            top_target.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "b",
        )
        self._place_inhand(self.plate_b, self._robot_manager_b, self.atom_b, top_target, "b")
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
        a_open, b_open, no_plate_welds, released = self._plates_released()
        stacked_upright = target_err < 0.04 and stack_err < 0.03 and height_err < 0.015 and a_up and b_up
        if stacked_upright and released:
            self._success_hold_count = getattr(self, "_success_hold_count", 0) + 1
        else:
            self._success_hold_count = 0

        self.metadata["plate_a"] = [float(v) for v in ap.p]
        self.metadata["plate_b"] = [float(v) for v in bp.p]
        self.metadata["target_err"] = target_err
        self.metadata["stack_err"] = stack_err
        self.metadata["height_err"] = height_err
        self.metadata["gripper_a_open"] = a_open
        self.metadata["gripper_b_open"] = b_open
        self.metadata["plates_unwelded"] = no_plate_welds
        self.metadata["released"] = released
        self.metadata["stacked_upright"] = bool(stacked_upright)
        self.metadata["success_hold_count"] = int(self._success_hold_count)
        print(
            f"[PLATE_STACK] target_err={target_err*1000:.1f}mm "
            f"stack_err={stack_err*1000:.1f}mm dz={dz*1000:.1f}mm "
            f"height_err={height_err*1000:.1f}mm a_up={a_up} b_up={b_up} "
            f"a_open={a_open:.2f} b_open={b_open:.2f} unwelded={no_plate_welds} "
            f"stacked_upright={stacked_upright} hold={self._success_hold_count}/{SUCCESS_HOLD_STEPS}",
            flush=True,
        )
        # 竖壁盘嵌套深度不固定 -> 用范围判"上盘坐在下盘上方"(去 weld 后套叠更自然)。
        nested = 0.012 < dz < 0.050
        return bool(target_err < 0.04 and stack_err < 0.03 and nested and a_up and b_up)
