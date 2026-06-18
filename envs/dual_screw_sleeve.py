from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Dual Screw-Sleeve —— 双臂装配: 左臂(A)举套筒, 右臂(B)把螺丝插进套筒 (资产 3x 放大)
#   screw : SCREW.usd  (图钉状: 杆 ⌀30×105 + 盘 ⌀72×12; 杆沿 local Z, 杆尖 -Z, 盘 +Z)
#   sleeve: SLEEVE.usd (空心圆柱: 外⌀54/内⌀36/长66; 孔轴 local Z, 上下通)
#   初始摆位: 螺丝盘朝上立在桌上(杆朝下); 套筒打横躺在桌上(孔轴水平)。
#   流程: A 抓起横躺套筒 -> 竖起举到装配位(孔朝上)保持 ; B 抓起螺丝(夹盘, 杆朝下) ->
#         对准套筒孔 -> 下压插入(带触觉)。
#   规划忽略: A 忽略在手 sleeve; B 忽略在手 screw + 要穿过的 sleeve(孔小, 体素化当实心)。
# ----------------------------------------------------------------------------
#   SCREW 局部(原点居中, z∈[-0.0578,0.0578]): 杆尖 local -Z, 盘 local +Z。原点->杆尖 = 0.0578m
#   SLEEVE 局部(原点居中, z∈[-0.033,0.033], 外径 0.054): 孔轴 local Z, 半长 0.033m
# ============================================================================

# ---- 几何 (3x, m) ----
SLEEVE_H     = 0.044    # 套筒长(孔上下通)  [资产已缩到 2/3]
SLEEVE_OUT_R = 0.018    # 套筒外半径(横躺时贴桌高度)
SCREW_TIP    = 0.0385   # 螺丝原点 -> 杆尖(local -Z)
INSERT_DEPTH = 0.055    # [tune] 杆插入套筒深度(加大, 插深点)
SLEEVE_GRASP_AXIS = 0.0    # 抓套筒沿轴向偏移(0=体心; 之前 +Y 偏移会让抓取/举起失败, 暂回退)

# ---- 摆位 [tune] (世界系) ----
# 物体放在各自臂正前方(与基座同 y), 让 top-down 抓取等价于已验证的单臂抓取(基座原点->(0.5,0,桌面))。
LYING_QUAT = [0.70711, 0.70711, 0, 0]   # 绕 X 转 90°: 孔轴 local Z -> 世界 -Y(套筒打横躺)
# 螺丝: 盘朝上立桌(identity), 杆尖贴桌 -> 体心 z = 半高 + 余隙; 在 arm B 正前方
SCREW_REST    = Pose([0.50, -0.35, 0.042], [1, 0, 0, 0])
# 套筒: 打横躺; 在 arm A 正前方
SLEEVE_REST   = Pose([0.50,  0.35, 0.021], LYING_QUAT)
# A 把套筒举到此, 并加"朝外"倾角(孔口朝 arm B/略下), 螺丝按此角度插入。
# 用 _place_inhand 直接算夹爪目标摆放(不走 place_actor, 故倾斜也能摆正)。
# [tune] world euler(rad): 绕 X 下倾, 绕 Z 转向 arm B(-x)。
ASSEMBLY_TILT = [0.0, 0.0, -0.3]   # 绕竖直轴 yaw 把孔口转向外侧(~17°, 比之前小), 不下倾
ASSEMBLY_POSE = Pose([0.48,  0.18, 0.22], LYING_QUAT).add_rotation(ASSEMBLY_TILT, coord='world')   # y 挪近 arm A

# ---- 抓取 [tune] ----
SLEEVE_GRASP_DZ = 0.0      # 抓横躺套筒: 顶面下扎(沿外径中心)
SCREW_GRASP_DZ  = 0.033    # 抓螺丝: 体心往 +Z(盘端)偏, 顶夹圆盘, 杆悬在下方


@configclass
class TaskCfg(BaseTaskCfg):
    dual_arm = True       # <- 开启双臂(自动建 arm B: /Robot_b + 触觉 *_b)
    # GUI 默认相机: 从前上方俯看整个工作区(两臂 + 桌面), 替换 base 那个又低又偏的默认视角。
    viewer = ViewerCfg(eye=(1.45, 0.0, 0.95), lookat=(0.45, 0.0, 0.12))
    video_size = (1760, 320)   # head + wrist + wrist_b (3×480) + 4 触觉(2列×160) = 1760 宽
    # 双臂工作区更宽: 桌面放大到 2×2m, 覆盖两条臂(基座 y=±0.40)。厚度不变(top 仍 z=0.0025)。
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
        cfg.uipc_sim.contact.default_friction_ratio = 2.5   # 摩擦回中(高摩擦+深夹会让 UIPC 接触求解很慢=举起卡); 夹持改靠 reassert 冻结关节
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {'sleeve'}            # A 手里的套筒
        self._robot_manager_b.ignore_actors = {'screw', 'sleeve'}  # B 手里的螺丝 + 要穿过的套筒

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        self.sleeve = self._actor_manager.add_from_usd_file(
            name='sleeve', asset_path="SLEEVE.usd", pose=SLEEVE_REST)
        self.screw = self._actor_manager.add_from_usd_file(
            name='screw', asset_path="SCREW.usd", pose=SCREW_REST)

    def _reset_actors(self):
        self.sleeve.set_pose(SLEEVE_REST.add_offset(self.create_noise([0.005, 0.005, 0.0])))
        self.screw.set_pose(SCREW_REST.add_offset(self.create_noise([0.005, 0.005, 0.0])))

    # ---------------------------------------------------------------- helpers
    def _grasp(self, actor, atom, arm, dz, grasp_from=(0, 0, 1), camera_up=(1, 0, 0),
               extra_bias=(0.0, 0.0, 0.0), grip_depth=28.0, open_amt=1.0):
        """抓取, 走指定臂(construct_grasp_pose 范式)。自适应闭合到触觉深度 grip_depth
        (停在刚接触+适度压的正确开度, 不会像固定开度那样过穿透把物体挤飞; 越小夹越深越紧)。
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
        绕开 place_actor 的"对齐"语义(对带倾角的目标会摆歪)。"""
        inhand = actor.get_pose().rebase(rm.get_gripper_center_pose())          # actor 在夹爪系下
        gc_mat = np.array(target_pose.to_transformation_matrix()) @ \
            np.linalg.inv(np.array(inhand.to_transformation_matrix()))          # 夹爪中心目标
        ee = rm.gripper_center_to_ee(Pose.from_matrix(gc_mat))
        self.move(atom.move_to_pose(ee), arm=arm)

    def _hole_axis(self):
        """套筒孔轴(世界系), 取指向 arm B(-Y 侧)的那一端方向。跟随套筒实际朝向, 支持装配位带倾角。"""
        sp = self.sleeve.get_pose()
        axis = sp.to_transformation_matrix()[:3, 2]          # 套筒 local Z = 孔轴(世界)
        if axis[1] > 0:
            axis = -axis                                     # 指向 -Y(朝 arm B 外侧)
        return np.array(axis, dtype=float)

    def _hole_pose(self):
        """螺丝插入目标(螺丝原点目标): 朝向=套筒实际朝向(螺丝杆 local -Z 自动指入孔),
        原点在朝 arm B 那侧孔口外 SCREW_TIP 处。装配位倾斜时自动跟随。"""
        sp = self.sleeve.get_pose()
        axis = self._hole_axis()
        origin = np.array(sp.p, dtype=float) + axis * (SLEEVE_H / 2 + SCREW_TIP)   # 孔口外
        return Pose(origin.tolist(), sp.q)

    def _dbg(self, tag):
        ga = self._robot_manager.get_gripper_qpos()        # arm A 夹爪开度(qpos)
        sp = self.sleeve.get_pose()
        print(f"[DBG] {tag}: armA_grip_qpos={ga:.4f}  sleeve=({sp.p[0]:.3f},{sp.p[1]:.3f},{sp.p[2]:.3f})", flush=True)

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(10)   # 仅让两个物体落稳; 抓取/插入全放到 _play_once(会被录进视频)

    def _play_once(self):
        # A: 抓起横躺套筒(顶面下扎, 两指跨过 ⌀54 管) -> 横着举到装配位保持(水平, 孔轴沿 Y)
        # camera_up=(0,-1,0): 手指开合方向仍=世界 X(夹左右筒壁、不堵 ±Y 孔), 但绕接近轴翻 180°,
        # 让 panda 腕关节换个姿态(approach 更顺、夹得更实)。grip_depth 夹紧防圆筒打滑。
        # 自适应闭合(深度 28=适度压紧, 停在正确开度不过穿透); reassert 冻结保持
        self._grasp(self.sleeve, self.atom_a, 'a', SLEEVE_GRASP_DZ, camera_up=(0, -1, 0),
                    extra_bias=(0.0, SLEEVE_GRASP_AXIS, 0.0), grip_depth=28.0, open_amt=0.55)
        self._place_inhand(self.sleeve, self._robot_manager, self.atom_a, ASSEMBLY_POSE, 'a')
        self._dbg("A 举好套筒, B 还没动")
        # B: 自适应夹螺丝圆盘(杆悬在下方), 抬离桌面
        self._grasp(self.screw, self.atom_b, 'b', SCREW_GRASP_DZ, camera_up=(1, 0, 0), grip_depth=28.0)
        self._dbg("B 夹螺丝后")
        self.move(self.atom_b.move_by_displacement(z=0.15, xyz_coord='world'), arm='b'); self._dbg("B 抬螺丝后")

        # B: 横向插入 —— 螺丝杆转到水平(+Y), 从 -Y 侧对准孔口, 再 +Y 推入
        sp = self.sleeve.get_pose()
        print(f"[DBG] 套筒插入前 pose = p({sp.p[0]:.3f},{sp.p[1]:.3f},{sp.p[2]:.3f}) "
              f"q({sp.q[0]:.3f},{sp.q[1]:.3f},{sp.q[2]:.3f},{sp.q[3]:.3f})  "
              f"(装配位应为 0.48,0.10,0.22)", flush=True)
        hole = self._hole_pose()
        axis = self._hole_axis()                             # 孔轴(朝 arm B 外侧)
        hover = Pose((np.array(hole.p) + axis * 0.08).tolist(), hole.q)   # 沿孔轴往外 8cm 悬停
        self._place_inhand(self.screw, self._robot_manager_b, self.atom_b, hover, 'b')
        self._place_inhand(self.screw, self._robot_manager_b, self.atom_b, hole, 'b')   # 对准孔口
        # 直线插入: 沿夹爪局部 z(对准后正好=插入方向)走 + constraint_pose 锁住其余5轴 -> curobo 解出直线
        # (和 insert_USB 同款; 之前用世界 Y+错的约束所以解不出、也不直)
        self.move(self.atom_b.move_by_displacement(z=INSERT_DEPTH * 0.6, xyz_coord='local'),
                  arm='b', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(self.atom_b.move_by_displacement(z=INSERT_DEPTH * 0.5, xyz_coord='local'),
                  arm='b', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.delay(20, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        screw_pose = self.screw.get_pose()
        sleeve_pose = self.sleeve.get_pose()
        rel = screw_pose.rebase(sleeve_pose)             # 螺丝在套筒坐标系下(孔轴=套筒 local Z)
        self.metadata['rel_pose'] = rel.tolist()
        # 杆轴与孔轴共线(横向插入, 用 abs): 螺丝 local Z 在套筒系应 ≈ ±Z
        axis_dot = float(np.dot(rel.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])))
        axis_aligned = abs(axis_dot) > 0.9
        radial = float(np.linalg.norm(rel.p[0:2]))       # 垂直孔轴方向的偏移
        radial_ok = radial < 0.008
        depth_ok = bool(abs(rel.p[2]) < SLEEVE_H / 2 + 0.02)  # 沿孔轴落在套筒内
        print(f"[DUAL] rel(套筒系) radial={radial*1000:.1f}mm along_axis={rel.p[2]*1000:.1f}mm "
              f"|axis_dot|={abs(axis_dot):.3f} -> aligned={axis_aligned} radial={radial_ok} depth={depth_ok}", flush=True)
        return bool(axis_aligned and radial_ok and depth_ok)
