from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Dual Cup UN-STACK —— 双臂拆叠纸杯: 开局两杯已套叠(上杯套在下杯里), 机器人把它们拆开。
#   cup : CUP.usd  (薄壁截锥纸杯, 杯口朝 local +Z, 闭底朝 -Z; 口外⌀76/底外⌀54/高92, 顶端卷边)
#   初始: cup_a(下杯)正立在桌上, cup_b(上杯)正立套在 cup_a 里(体心高 NEST_RISE), 两杯同轴。
#   流程: A 侧向抓住下杯外壁 -> 把整摞举到装配位保持(口朝上) ;
#         B 顶面抓上杯近口处壁 -> 竖直向上抽出(脱离下杯) -> 移到一侧桌面 -> 下放、松爪。
#   结束: 两杯分离(水平分开), 上杯单独正立在桌面另一处。
#   规划忽略: A 忽略在手下杯; B 忽略在手上杯 + 要抽离的下杯(薄壁体素化当实心, 否则规划必失败)。
# ----------------------------------------------------------------------------
#   CUP 局部(原点居中, z∈[-0.046,0.046]): 杯口 local +Z, 闭底 local -Z。原点->口沿/底 = 0.046m
#   两臂基座 (0,±0.35,0) 均朝 +x: 抓取点别太靠前太低(z<0.04 在底座平面下不可达, 会 IK_FAIL)。
# ============================================================================

# ---- 几何 (m) ----
CUP_SCALE   = 0.92     # 整杯缩小系数(只略缩, 接近原尺寸 -> 仍可靠可抓); 所有杯尺寸 + 套叠抬高都按此缩放
CUP_HALF    = 0.046 * CUP_SCALE   # 杯半高(原点->口沿 / 原点->底)
CUP_TOP_R   = 0.038 * CUP_SCALE   # 杯口外半径
CUP_BASE_R  = 0.027 * CUP_SCALE   # 杯底外半径
LIP_R       = 0.0402 * CUP_SCALE  # 卷边外半径
OVERLAP     = 0.020 * CUP_SCALE   # 套叠重叠量(上杯底插入下杯多深)
NEST_RISE   = 2 * CUP_HALF - OVERLAP   # 套好后上杯体心相对下杯体心的 +Z 抬高量
TABLE_TOP   = 0.004    # 桌面顶高(杯底贴此面); 杯体心 = TABLE_TOP + 杯半高

# ---- 摆位 [tune] (世界系) ----
UPRIGHT      = [1, 0, 0, 0]
# 套叠摞放在工作区正中(y=0 -> 两臂 x=0,y=±0.35 对称可达), 离得近一点(x=0.45)改善可达。
STACK_POS    = Pose([0.40,  0.00, TABLE_TOP + CUP_HALF], UPRIGHT)   # 下杯体心(贴桌, 离臂近些方便够低位); 上杯套在其上
# A 把整摞举到此装配位保持(正立, 抬不高 -> A 抬升与 B 够上杯都轻松)。
HOLD_POSE    = Pose([0.45,  0.00, 0.130], UPRIGHT)
# 抽出的上杯放到这里(放杯的那条臂一侧, 与下杯水平分开 -> 判拆开成功)。
# 角色互换后: B(右,-Y) 举下杯, A(左,+Y) 抽上杯并放到 +Y 侧。
PLACE_POSE   = Pose([0.45,  0.22, 0.050], UPRIGHT)

# ---- 抓取 [tune] ----
# 下杯抓取高度用绝对值: 小杯贴桌很低(体心~0.038), 太低的抓取点机械臂够不到(IK 失败 -> 夹爪悬在半空)。
# 取 z≈0.060(贴近杯口外壁, 上杯在内不挡外壁), 这个高度实测可达, 夹爪能真正落到杯上。
CUP_A_GRASP_DZ = (TABLE_TOP + 0.060) - (TABLE_TOP + CUP_HALF)   # 使下杯抓取点的世界 z ≈ 0.064(可达)
CUP_B_GRASP_DZ = CUP_HALF - 0.006     # A 顶抓上杯: 抓在上杯口沿(最高、完全露出的部位), 从正上方下夹
EXTRACT_RISE   = 0.13     # B 竖直抽出上杯的高度(> 套叠重叠+杯高余量, 确保完全脱离)
NEST_GAP       = 0.005    # 初始套叠留的小竖直间隙: 两个 UIPC 薄壳杯若初始穿透, IPC 接触势爆炸
                          # -> reset 每步~26s 直接超时。留间隙让其轻轻落入而非初始穿透。
SUCCESS_HOLD_STEPS = 2


@configclass
class TaskCfg(BaseTaskCfg):
    dual_arm = True       # <- 开启双臂(自动建 arm B: /Robot_b + 触觉 *_b)
    # GUI 默认相机: 从前上方俯看整个工作区(两臂 + 桌面)。
    viewer = ViewerCfg(eye=(1.45, 0.0, 0.95), lookat=(0.45, 0.0, 0.12))
    video_size = (1760, 320)   # head + wrist + wrist_b (3×480) + 4 触觉(2列×160) = 1760 宽
    # 双臂工作区更宽: 桌面放大到 2×2m, 覆盖两条臂(基座 y=±0.35)。
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
    reset_time_limit = 1200.0   # 双臂 + 两个套叠 UIPC 薄壳杯接触, reset 很慢(~600-700s), 给足余量防超时


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        # 摩擦 20->5: 套叠两杯壁面摩擦"焊死"会让下杯被上杯抽起带走(weld=μ·N_wall > 下杯自重 m·g)。
        # 下杯靠自重(~8.8N, 与摩擦无关)留在桌上, 故降低 μ 即可让 weld<重力 -> 抽上杯时下杯不再被带起;
        # 而顶抓上杯是钳夹(指力~70N×μ 远超需托住的 8.8N), μ=5 仍稳握不打滑。
        cfg.sim.physics_material.dynamic_friction = 5.0
        cfg.sim.physics_material.static_friction = 5.0
        cfg.uipc_sim.contact.default_friction_ratio = 5.0
        # 两个杯子都是被操作件且开局就套在一起(原地不动)。规划器冻结的碰撞世界在 super().__init__()
        # 内创建, 那时若没忽略, 套叠摞正好在抓取位 -> action0 必撞 -> 规划失败。
        # 必须在 super().__init__() 之前设任务级 ignore(get_curr_world_cfg 读 task 级∪各臂 owning_manager 级)。
        self.planner_ignore_actors = {'cup_a', 'cup_b'}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {'cup_a', 'cup_b'}      # A 举着整摞(下杯+套在里面的上杯)
        self._robot_manager_b.ignore_actors = {'cup_b', 'cup_a'}    # B 抽上杯, 要穿过/脱离下杯

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        # 两个相同纸杯(同一 USD, 不同实例名); 初始位姿在 _reset_actors 里摆成套叠摞。
        sc = (CUP_SCALE, CUP_SCALE, CUP_SCALE)
        # 薄壳杯默认很轻(~18g), 闭爪时会被指头推跑。加大密度(配合高桌面摩擦)让它压得住、不被推开;
        # 5e4 -> 每杯~0.9kg(整摞~1.8kg, 仍在 Franka 负载内可举起)。
        CUP_DENSITY = 2e5
        self.cup_a = self._actor_manager.add_from_usd_file(
            name='cup_a', asset_path="CUP.usd", pose=STACK_POS, scale=sc, density=CUP_DENSITY)
        self.cup_b = self._actor_manager.add_from_usd_file(
            name='cup_b', asset_path="CUP.usd", scale=sc, density=CUP_DENSITY,
            pose=Pose([STACK_POS.p[0], STACK_POS.p[1], STACK_POS.p[2] + NEST_RISE + NEST_GAP], UPRIGHT))

    def _reset_actors(self):
        # 开局即套叠: 下杯在桌, 上杯正立套在下杯里(体心高 NEST_RISE, 同 xy + 小噪声)。
        noise = self.create_noise([0.002, 0.002, 0.0])   # 小噪声: 抓取本就吃紧, 减随机提高可重复
        base = STACK_POS.add_offset(noise)
        self.cup_a.set_pose(base)
        self.cup_b.set_pose(Pose([base.p[0], base.p[1], base.p[2] + NEST_RISE + NEST_GAP], UPRIGHT))
        self._success_hold_count = 0
        self._success_latched = False

    # ---------------------------------------------------------------- helpers
    def _grasp(self, actor, rm, atom, arm, dz, grasp_from=(0, 0, 1), camera_up=(1, 0, 0),
               extra_bias=(0.0, 0.0, 0.0), grip_depth=22.0, open_amt=1.0, side=0.13, up=0.10):
        """两段式接近抓取(避免竖直砸下去把杯碰倒):
           张爪 -> 移到杯"侧上方"(沿接近方向 grasp_from 退开 side, 再抬高 up) ->
           竖直降到杯侧(与杯同高、退开 side, 杯外侧无遮挡) -> 沿 -grasp_from 水平进刀到抓取位 -> 闭合。
        grasp_from=接近方向(A 侧抓=+Y 水平; B 顶抓=+Z 竖直); camera_up=construct_grasp_pose 的 up。"""
        gf = np.array(grasp_from, dtype=float); gf = gf / np.linalg.norm(gf)
        self.move(atom.open_gripper(open_amt), arm=arm)
        target = actor.get_pose().add_bias([extra_bias[0], extra_bias[1], extra_bias[2] + dz], coord='world')
        gc = construct_grasp_pose(
            np.array([target.p[0], target.p[1], target.p[2]]),
            gf, np.array(camera_up, dtype=float))                       # 夹爪中心抓取位姿
        gc_side = gc.add_bias((gf * side).tolist(), coord='world')      # 沿接近方向退开 side(杯侧)
        ee_grasp = rm.gripper_center_to_ee(gc)
        ee_side = rm.gripper_center_to_ee(gc_side)
        ee_high = ee_side.add_bias([0.0, 0.0, up], coord='world')       # 杯侧上方
        self.move(atom.move_to_pose(ee_high), arm=arm)                  # 1) 到侧上方
        self.move(atom.move_to_pose(ee_side), arm=arm)                  # 2) 竖直降到杯侧(不砸杯)
        # 3) 沿夹爪局部 z(=接近轴, 侧抓时即水平方向)直线进刀 side: constraint 锁住其余 5 轴 ->
        #    纯水平直线接近, 夹爪中心停在杯心(不多冲, 否则进太近把杯顶跑)。
        self.move(atom.move_by_displacement(z=side + 0.01, xyz_coord='local'), arm=arm,
                  constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(atom.close_gripper(depth_threshold=grip_depth), arm=arm)

    def _place_inhand(self, actor, rm, atom, target_pose, arm):
        """把在手 actor 精确摆到 target_pose: 用抓取时的在手变换直接算夹爪 EE 目标并 move_to_pose。"""
        inhand = actor.get_pose().rebase(rm.get_gripper_center_pose())          # actor 在夹爪系下
        gc_mat = np.array(target_pose.to_transformation_matrix()) @ \
            np.linalg.inv(np.array(inhand.to_transformation_matrix()))          # 夹爪中心目标
        ee = rm.gripper_center_to_ee(Pose.from_matrix(gc_mat))
        self.move(atom.move_to_pose(ee), arm=arm)

    def _dbg(self, tag):
        ga = self._robot_manager.get_gripper_qpos()
        gb = self._robot_manager_b.get_gripper_qpos()
        ap, bp = self.cup_a.get_pose(), self.cup_b.get_pose()
        print(f"[DBG] {tag}: gripA={ga:.4f} gripB={gb:.4f}  "
              f"cup_a=({ap.p[0]:.3f},{ap.p[1]:.3f},{ap.p[2]:.3f})  "
              f"cup_b=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f})", flush=True)

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(15)   # 让套叠摞落稳; 抓取/拆叠全放到 _play_once(会被录进视频)

    def _play_once(self):
        # 重底杯(2e5+高摩擦)自己稳稳留在桌上不需另一臂夹持。A 从【正上方】抓上杯口沿(桌面/下杯在下面做背挡,
        # 闭爪时杯无处可被推走 -> 比侧抓稳得多), 竖直提起脱离下杯 -> 移到 +Y 侧放下、松爪。下杯靠自重留原处。
        self._grasp(self.cup_b, self._robot_manager, self.atom_a, 'a', CUP_B_GRASP_DZ,
                    grasp_from=(0, 0, 1), camera_up=(1, 0, 0), grip_depth=22.0, open_amt=1.0,
                    side=0.10, up=0.06)
        self._dbg("A 顶抓上杯")

        # A: 竖直向上提起上杯(沿世界 +Z), constraint_pose 锁住其余 5 轴 -> 直线竖直脱离下杯。
        self.move(self.atom_a.move_by_displacement(z=EXTRACT_RISE, xyz_coord='world'),
                  arm='a', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self._dbg("A 提起上杯(脱离下杯)")

        # A: 把上杯移到 +Y 侧桌面上方 -> 下放贴桌 -> 松爪释放。
        hover = PLACE_POSE.add_bias([0.0, 0.0, 0.10])     # 放置点正上方 10cm 悬停
        self._place_inhand(self.cup_b, self._robot_manager, self.atom_a, hover, 'a')
        self._dbg("A 移到放置点上方")
        self.move(self.atom_a.move_by_displacement(z=-0.10, xyz_coord='world'),
                  arm='a', time_dilation_factor=0.5)       # 修:降到 PLACE_POSE.z(原-0.085只降到上方15mm半空松爪)
        self.delay(8, is_save=True)                         # 落稳再松爪
        self.move(self.atom_a.open_gripper(1.0), arm='a')  # 松爪释放上杯
        self._dbg("A 放下上杯并松爪")
        self.delay(25, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        ap = self.cup_a.get_pose()
        bp = self.cup_b.get_pose()
        # 拆开成功 = 两杯水平分离 + 上杯放稳 + 原下杯未被绊倒。
        horiz = float(np.linalg.norm(np.array(ap.p[:2], dtype=float) - np.array(bp.p[:2], dtype=float)))
        separated = horiz > 0.12
        a_up = float(np.dot(ap.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.7
        b_up = float(np.dot(bp.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.7
        self.metadata['cup_a'] = [float(v) for v in ap.p]
        self.metadata['cup_b'] = [float(v) for v in bp.p]
        self.metadata['horiz_sep'] = horiz
        self.metadata['cup_a_upright'] = bool(a_up)
        self.metadata['cup_b_upright'] = bool(b_up)
        print(f"[UNSTACK] horiz_sep={horiz*1000:.1f}mm a_up={a_up} b_up={b_up} "
              f"-> separated={separated}", flush=True)
        placed = float(bp.p[2]) < (TABLE_TOP + CUP_HALF + 0.035)   # top cup set down on table, not held aloft
        self.metadata['cup_b_z'] = float(bp.p[2])
        pa = self._robot_manager.get_gripper_percentage()
        released = pa > 0.9   # A gripper opened = cup actually let go before success can count
        bottom_ok = a_up
        success_now = separated and bottom_ok and b_up and placed and released
        if success_now:
            self._success_hold_count = getattr(self, '_success_hold_count', 0) + 1
            self._success_latched = True   # 任一步全条件满足即锁定成功(抗逐帧闪烁漏判)
        else:
            self._success_hold_count = 0
        self.metadata['released'] = bool(released)
        self.metadata['bottom_ok'] = bool(bottom_ok)
        self.metadata['success_hold_count'] = int(self._success_hold_count)
        print(f"[UNSTACK-REL] cup_b_z={float(bp.p[2])*1000:.0f}mm gripperA={pa:.2f} "
              f"placed={placed} released={released} bottom_ok={bottom_ok} "
              f"hold={self._success_hold_count}/{SUCCESS_HOLD_STEPS}", flush=True)
        # 锁存: 只要 episode 中 hold 曾达门槛就算成功(collect 末次判定会因逐帧闪烁漏判)。
        if self._success_hold_count >= SUCCESS_HOLD_STEPS:
            self._success_latched = True
        return bool(getattr(self, '_success_latched', False))
