from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Dual Cup Stack —— 双臂套叠纸杯: 左臂(A)举住下杯(承接), 右臂(B)抓上杯套进下杯
#   cup : CUP.usd  (薄壁截锥纸杯, 杯口朝 local +Z, 闭底朝 -Z; 口外⌀76/底外⌀54/高92, 顶端卷边)
#   初始摆位: 两个相同纸杯都正立(口朝上)在桌上, 各在一条臂正前方。
#   流程: A 侧向抓住下杯(夹两侧壁, 杯口 +Z 不挡) -> 举到装配位保持(口朝上) ;
#         B 顶面抓上杯(夹近口处壁, 底悬在下方) -> 抬起 -> 移到下杯口正上方 ->
#         底先入、竖直下压套入(带触觉; 卷边卡住=套叠到位)。
#   规划忽略: A 忽略在手下杯; B 忽略在手上杯 + 要套入的下杯(薄壁体素化当实心, 否则规划必失败)。
# ----------------------------------------------------------------------------
#   CUP 局部(原点居中, z∈[-0.046,0.046]): 杯口 local +Z, 闭底 local -Z。原点->口沿/底 = 0.046m
#   (几何来自 scripts/asset_tools/build_cup.py, 单位 m)
# ============================================================================

# ---- 几何 (m) ----
CUP_HALF    = 0.046    # 杯半高(原点->口沿 / 原点->底)
CUP_TOP_R   = 0.038    # 杯口外半径
CUP_BASE_R  = 0.027    # 杯底外半径
LIP_R       = 0.0402   # 卷边外半径(套叠时卡住的位置)
OVERLAP     = 0.020    # [tune] 套叠重叠量(卷边卡住前上杯底插入下杯多深)
NEST_RISE   = 2 * CUP_HALF - OVERLAP   # 套好后上杯体心相对下杯体心的 +Z 抬高量(≈0.072)

# ---- 摆位 [tune] (世界系) ----
# 两杯都正立(口朝上, identity), 各放在对应臂正前方(与基座同 y); 底贴桌 -> 体心 z = 半高 + 余隙。
UPRIGHT      = [1, 0, 0, 0]
CUP_A_REST   = Pose([0.50,  0.30, 0.050], UPRIGHT)   # 下杯(承接): arm A 正前方(+Y)
CUP_B_REST   = Pose([0.50, -0.30, 0.050], UPRIGHT)   # 上杯(套入): arm B 正前方(-Y)
# A 把下杯举到此装配位保持(正立, 口朝上, 略挪向中线方便 B 从上方套)。
HOLD_POSE    = Pose([0.50,  0.12, 0.170], UPRIGHT)

# ---- 抓取 [tune] ----
CUP_A_GRASP_DZ = -0.010   # A 侧抓下杯: 体心略下(夹中下段壁), 让杯口 +Z 段空出给 B 套入
CUP_B_GRASP_DZ =  0.034   # B 顶抓上杯: 体心往 +Z(口端)偏, 夹近口沿处壁, 杯底悬在下方


@configclass
class TaskCfg(BaseTaskCfg):
    dual_arm = True       # <- 开启双臂(自动建 arm B: /Robot_b + 触觉 *_b)
    # GUI 默认相机: 从前上方俯看整个工作区(两臂 + 桌面)。
    viewer = ViewerCfg(eye=(1.45, 0.0, 0.95), lookat=(0.45, 0.0, 0.12))
    video_size = (1760, 320)   # head + wrist + wrist_b (3×480) + 4 触觉(2列×160) = 1760 宽
    # 双臂工作区更宽: 桌面放大到 2×2m, 覆盖两条臂(基座 y=±0.40)。
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
            offset=CameraCfg.OffsetCfg(pos=(1.35, 0.0, 0.85), rot=(0.6352, 0.3107, 0.3107, 0.6352), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
            ),
            width=480, height=270, update_period=1/120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480, height=270, update_period=1/120,
        ),
        CameraCfg(
            name="wrist_b",
            prim_path="/World/envs/env_.*/Robot_b/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480, height=270, update_period=1/120,
        ),
    ]
    step_lim = 1500
    reset_time_limit = 600.0   # 双臂场景重(2 套 curobo + 4 触觉 + 渲染 + 高摩擦), reset 慢, 给足余量


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5   # 高摩擦防薄壁打滑; 夹持靠 reassert 冻结关节
        # 两个杯子都是被操作件(不像 lift_can 的罐子会被挪到远处, 杯子原地不动), 规划器冻结的碰撞世界
        # 在 super().__init__() 内创建, 那时若没忽略, cup_a 正好在 arm A 抓取位 -> action0 必撞 -> 规划失败。
        # 必须在 super().__init__() 之前设任务级 ignore(get_curr_world_cfg 读 task 级∪各臂 owning_manager 级)。
        self.planner_ignore_actors = {'cup_a', 'cup_b'}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {'cup_a'}              # A 手里的下杯
        self._robot_manager_b.ignore_actors = {'cup_b', 'cup_a'}    # B 手里的上杯 + 要套入的下杯

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        # 两个相同纸杯(同一 USD, 不同实例名)
        self.cup_a = self._actor_manager.add_from_usd_file(
            name='cup_a', asset_path="CUP.usd", pose=CUP_A_REST)
        self.cup_b = self._actor_manager.add_from_usd_file(
            name='cup_b', asset_path="CUP.usd", pose=CUP_B_REST)

    def _reset_actors(self):
        self.cup_a.set_pose(CUP_A_REST.add_offset(self.create_noise([0.005, 0.005, 0.0])))
        self.cup_b.set_pose(CUP_B_REST.add_offset(self.create_noise([0.005, 0.005, 0.0])))

    # ---------------------------------------------------------------- helpers
    def _grasp(self, actor, atom, arm, dz, grasp_from=(0, 0, 1), camera_up=(1, 0, 0),
               extra_bias=(0.0, 0.0, 0.0), grip_depth=28.0, open_amt=1.0):
        """抓取, 走指定臂(construct_grasp_pose 范式)。自适应闭合到触觉深度 grip_depth
        (停在刚接触+适度压的正确开度, 不会过穿透把物体挤飞; 越小夹越深越紧)。
        grasp_from = 接近方向; camera_up = construct_grasp_pose 的 up(panda 沿 EE-Y 开合);
        dz/extra_bias = 抓取点偏置; open_amt = 抓前张开量。"""
        self.move(atom.open_gripper(open_amt), arm=arm)
        target = actor.get_pose().add_bias([extra_bias[0], extra_bias[1], extra_bias[2] + dz], coord='world')
        cpose = construct_grasp_pose(
            np.array([target.p[0], target.p[1], target.p[2]]),
            np.array(grasp_from, dtype=float), np.array(camera_up, dtype=float))
        cid = actor.register_point(cpose, type='contact')
        self.move(atom.grasp_actor(actor, contact_point_id=cid, is_close=False), arm=arm)
        self.move(atom.close_gripper(depth_threshold=grip_depth), arm=arm)

    def _place_inhand(self, actor, rm, atom, target_pose, arm):
        """把在手 actor 精确摆到 target_pose: 用抓取时的在手变换直接算夹爪 EE 目标并 move_to_pose,
        绕开 place_actor 的"对齐"语义(对自定义朝向的目标更稳)。"""
        inhand = actor.get_pose().rebase(rm.get_gripper_center_pose())          # actor 在夹爪系下
        gc_mat = np.array(target_pose.to_transformation_matrix()) @ \
            np.linalg.inv(np.array(inhand.to_transformation_matrix()))          # 夹爪中心目标
        ee = rm.gripper_center_to_ee(Pose.from_matrix(gc_mat))
        self.move(atom.move_to_pose(ee), arm=arm)

    def _stack_pose(self, rise):
        """上杯套入目标(上杯原点目标): 朝向=下杯实际朝向(都正立), 原点在下杯体心正上方 rise 处。
        rise=NEST_RISE 即套叠到位; rise 更大=悬停在口上方。下杯被举着带噪声/倾斜也自动跟随。"""
        ap = self.cup_a.get_pose()
        up = ap.to_transformation_matrix()[:3, 2]            # 下杯 local +Z = 杯口轴(世界)
        origin = np.array(ap.p, dtype=float) + up * rise
        return Pose(origin.tolist(), ap.q)

    def _dbg(self, tag):
        gb = self._robot_manager_b.get_gripper_qpos()
        ap, bp = self.cup_a.get_pose(), self.cup_b.get_pose()
        print(f"[DBG] {tag}: armB_grip_qpos={gb:.4f}  "
              f"cup_a=({ap.p[0]:.3f},{ap.p[1]:.3f},{ap.p[2]:.3f})  "
              f"cup_b=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f})", flush=True)

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(10)   # 仅让两个杯子落稳; 抓取/套叠全放到 _play_once(会被录进视频)

    def _play_once(self):
        # A: 侧向抓下杯(grasp_from=+Y 水平接近, camera_up=+Z -> 两指沿世界 X 夹两侧壁, 杯口 +Z 不挡),
        #    举到装配位保持(正立, 口朝上)。grip_depth 夹紧防薄壁打滑; 自适应闭合停在正确开度。
        self._grasp(self.cup_a, self.atom_a, 'a', CUP_A_GRASP_DZ,
                    grasp_from=(0, 1, 0), camera_up=(0, 0, 1), grip_depth=28.0, open_amt=0.8)
        self._place_inhand(self.cup_a, self._robot_manager, self.atom_a, HOLD_POSE, 'a')
        self._dbg("A 举好下杯, B 还没动")

        # B: 顶面抓上杯(夹近口沿处壁, 杯底悬在下方), 抬离桌面
        self._grasp(self.cup_b, self.atom_b, 'b', CUP_B_GRASP_DZ,
                    grasp_from=(0, 0, 1), camera_up=(1, 0, 0), grip_depth=28.0, open_amt=0.8)
        self._dbg("B 夹上杯后")
        self.move(self.atom_b.move_by_displacement(z=0.15, xyz_coord='world'), arm='b')
        self._dbg("B 抬上杯后")

        # B: 移到下杯口正上方 -> 竖直下压套入
        ap = self.cup_a.get_pose()
        print(f"[DBG] 套入前下杯 pose = p({ap.p[0]:.3f},{ap.p[1]:.3f},{ap.p[2]:.3f}) "
              f"(装配位应为 {HOLD_POSE.p[0]:.2f},{HOLD_POSE.p[1]:.2f},{HOLD_POSE.p[2]:.2f})", flush=True)
        hover = self._stack_pose(NEST_RISE + 0.07)    # 下杯口正上方 7cm 悬停(同轴对准)
        self._place_inhand(self.cup_b, self._robot_manager_b, self.atom_b, hover, 'b')
        self._dbg("B 悬停对准下杯口上方")
        # 竖直下压套入: 沿夹爪局部 z(对准后正好=竖直向下) + constraint_pose 锁住其余 5 轴 -> curobo 解直线
        # (和 insert_USB / 双臂插螺丝同款约束)。分两段下压, 减速防薄壁失稳。
        self.move(self.atom_b.move_by_displacement(z=0.04, xyz_coord='local'),
                  arm='b', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self._dbg("B 下压第一段")
        self.move(self.atom_b.move_by_displacement(z=0.035, xyz_coord='local'),
                  arm='b', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self._dbg("B 下压到位(套叠)")
        self.delay(20, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        cup_a_pose = self.cup_a.get_pose()
        cup_b_pose = self.cup_b.get_pose()
        rel = cup_b_pose.rebase(cup_a_pose)              # 上杯在下杯坐标系下(下杯 local Z = 杯口轴)
        self.metadata['rel_pose'] = rel.tolist()
        # 两杯轴共线(都正立, 套叠): 上杯 local Z 在下杯系应 ≈ +Z
        axis_dot = float(np.dot(rel.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])))
        axis_aligned = axis_dot > 0.9
        radial = float(np.linalg.norm(rel.p[0:2]))       # 同轴(垂直杯口轴)偏移
        radial_ok = radial < 0.012
        # 沿轴: 上杯体心在下杯体心上方, 落在套叠范围(重叠->比 2*half 小, 但确实套进去了)
        along = float(rel.p[2])
        nested_ok = bool(0.045 < along < 2 * CUP_HALF + 0.005)
        print(f"[DUAL] rel(下杯系) radial={radial*1000:.1f}mm along_axis={along*1000:.1f}mm "
              f"axis_dot={axis_dot:.3f} -> aligned={axis_aligned} radial={radial_ok} nested={nested_ok}", flush=True)
        return bool(axis_aligned and radial_ok and nested_ok)
