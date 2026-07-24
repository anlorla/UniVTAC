from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Pour Ball —— 把杯子里的小球倒到盘子上。
#   单臂: 抓住装着球的杯子(cup) -> 举起 -> 移到盘子(plate)上方 -> 倾倒 -> 球落到盘上。
#   盘(plate)= 固定接收面(kinematic), 上面不放任何东西; 杯子是被抓取的动力学件。
#   杯用 CUP.usd(奶白纸杯, 杯口 local +Z, 闭底 local -Z; 居中, 口外⌀76/底外⌀54/高92)。
#   盘用 PLATE.usd(浅圆盘 Ø200, 边沿 8mm)。球用 BALL.usd(⌀16mm)。
# ----------------------------------------------------------------------------
#   相机/视角沿用 dual_screw_sleeve 的居中俯视范式(原来的偏置视角看着很怪)。
#   抓取改为杯身【侧抓】: 先到杯侧上方, 再竖直降到杯旁, 最后水平直线进刀闭合。
#   这样倾倒时手腕对杯有更大力臂, 比之前从正上方夹杯口更容易真正翻杯倒球。抓后 weld 杯到夹爪。
#   场景中心放在 (0.5, 0)(基座正前方), 与 screw 任务一致。
# ============================================================================

# ---- 几何 (m) ----
CUP_SCALE  = 0.93                    # 当前版本把杯放大到原先 1.5x(0.62 -> 0.93), 口外⌀~71mm, 接近夹爪开度上限;
                                     # 现改用侧抓而非顶抓, 主要依赖从杯身侧壁进刀而不是跨过杯口
CUP_HALF   = 0.046 * CUP_SCALE       # 杯半高(原点->口沿/底)
CUP_TOP_R  = 0.038 * CUP_SCALE       # 杯口外半径
BALL_R     = 0.008                   # 小球半径(BALL.usd, ⌀16mm)
TABLE_TOP  = 0.004                   # 桌面(ground_plate)顶高
PLATE_HALF_Z = 0.004                 # 盘半高
PLATE_CZ   = TABLE_TOP + PLATE_HALF_Z          # 盘心 z(盘底贴桌)
PLATE_R    = 0.10                    # 盘外半径(Ø200)

# ---- 摆位 [tune] (世界系) —— 场景中心在 (0.5,0), 同 screw 任务 ----
UPRIGHT      = [1, 0, 0, 0]
GAP          = 0.006                                          # 防 UIPC 初始穿透的小间隙
PLATE_POS    = Pose([0.50, -0.10, PLATE_CZ], UPRIGHT)         # 盘(接收面/倾倒目标) - 与杯交换位置
CUP_POS      = Pose([0.50,  0.10, TABLE_TOP + CUP_HALF + GAP], UPRIGHT)   # 杯(被抓, 内含球) - 与盘交换位置
BALL_DZ      = -CUP_HALF + 0.006 + BALL_R                     # 球相对杯心的 z(落在杯内底, 不穿壁)

# ---- 抓取 / 倾倒 [tune] ----
CUP_GRASP_DZ = 0.015                 # 杯放大后抓取点略上移, 仍夹在上半身靠口沿下方的杯壁
CUP_SIDE_APPROACH = 0.14             # 杯放大后, 侧抓时沿接近方向在杯外退开的距离也相应加大
CUP_SIDE_UP = 0.10                   # 侧抓预备位在杯侧上方抬高量
PREGRASP_CLEARANCE = 0.12            # 先原地竖直抬高再去预抓位, 避免斜线切向杯子把它碰倒
GRIP_CLOSE   = 0.60                  # 抓取闭合量(0=全闭/1=全开): 放松一些, 避免把大杯夹得过紧而挤形/扰动
LIFT_RISE    = 0.16                  # 保留作经验量级参考; 当前改为直接 move_to_pose 到倾倒预备位
POUR_OVER_Y  = 0.00                  # 倾倒位直接放到盘心上方, 不再保留之前那个人工偏置
POUR_Z       = 0.26                  # 倾倒位再抬高些, 给大杯+大角度翻转更多工作空间
POUR_PREP_X  = -0.03                 # 倾倒预备位略向机器人回收一点, 避免翻腕起始位太伸
POUR_ANGLE   = 210.0 / 180.0 * np.pi    # 过翻一些, 避免实际停在“接近倒扣但还不够”的姿态
POUR_LIFT    = 0.00                  # 倾倒阶段只翻腕不再同步抬高手位
POUR_JOINT_STEPS = 80                # 直接转 wrist joint 的插值步数


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 800
    use_adaptive_grasp = False  # 不用自适应抓取(它只夹到轻触阈值, 抓不牢杯子会滑/漏球); 改为直接 100% 闭合夹死
    reset_time_limit = 1200.0   # UIPC 接触求解偶发变慢(GPU 共用时); 给足余量
    # GUI 视角: 居中前上方俯视整个工作区(同 dual_screw_sleeve), 取代原来偏置的怪视角。
    viewer = ViewerCfg(eye=(1.45, 0.0, 0.95), lookat=(0.45, 0.0, 0.12))
    video_size = (1120, 320)   # head + wrist (2×480) + 2 触觉(1列×160) = 1120 宽
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
    ]


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        # 摩擦取中(2.0): μ=5 球粘在杯里倒不出; μ=0.8 球在翻杯途中过早滑出被甩飞到 -y 脱靶。
        # 取中等摩擦 + 把杯彻底翻过来(口朝下)over 盘心 -> 球在杯翻到口朝下时基本竖直落到盘上, 不被甩远。
        # 夹持靠 100% 闭合的几何夹死(不靠摩擦), 故放低摩擦不影响握杯。
        cfg.sim.physics_material.dynamic_friction = 2.0
        cfg.sim.physics_material.static_friction = 2.0
        cfg.uipc_sim.contact.default_friction_ratio = 2.0
        # 薄壳杯/盘/球被规划器体素化成实心障碍 -> 不忽略则 action0 必撞规划失败。忽略全部任务件。
        self.planner_ignore_actors = {'cup', 'ball', 'dish'}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {'cup', 'ball', 'dish'}

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        sc = (CUP_SCALE, CUP_SCALE, CUP_SCALE)
        # 盘: kinematic ABD(固定接收面)。self.plate 被基类占用(地面板), 故存为 self.dish。
        self.dish = self._actor_manager.add_from_usd_file(
            name='dish', asset_path="PLATE.usd", pose=PLATE_POS, density=1e4,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(kinematic=True))
        # 杯: 动力学件(被抓)。密度取真实纸杯量级(~5e3 -> 数十克), 不 weld 时自适应指力能稳稳夹住、
        # 倾倒时不因重力力矩在指间打滑(2e5 那种 ~0.9kg 会滑, 使在手变换漂移、后续规划目标算到够不到处)。
        self.cup = self._actor_manager.add_from_usd_file(
            name='cup', asset_path="CUP.usd", pose=CUP_POS, scale=sc, density=5e3)
        # 球: 动力学件, 初始在杯内底。
        self.ball = self._actor_manager.add_from_usd_file(
            name='ball', asset_path="BALL.usd",
            pose=Pose([CUP_POS.p[0], CUP_POS.p[1], CUP_POS.p[2] + BALL_DZ], UPRIGHT),
            density=2000.0)

    def _reset_actors(self):
        noise = self.create_noise([0.003, 0.003, 0.0])
        cup = CUP_POS.add_offset(noise)
        self.dish.set_pose(PLATE_POS)
        self.cup.set_pose(cup)
        self.ball.set_pose(Pose([cup.p[0], cup.p[1], cup.p[2] + BALL_DZ], UPRIGHT))
        self.metadata['plate_xy'] = [float(PLATE_POS.p[0]), float(PLATE_POS.p[1])]

    # ---------------------------------------------------------------- helpers
    def _grasp_cup(self):
        """从【侧面】抓杯上半身。

        轨迹采用两段式接近, 避免像顶抓那样直接压到杯口:
          1) 到杯侧上方悬停;
          2) 竖直降到杯侧;
          3) 沿夹爪局部 z(水平接近轴)直线进刀到杯心;
          4) 闭爪夹住杯身。
        """
        grasp_from = np.array([0.0, 1.0, 0.0], dtype=float)   # 恢复从 +Y 侧水平接近(旧方向)
        camera_up = np.array([0.0, 0.0, -1.0], dtype=float)   # 去掉不必要的 180° 翻腕; 指头仍沿 world-x 开合
        gf = grasp_from / np.linalg.norm(grasp_from)
        self.move(self.atom.open_gripper(1.0))
        self.move(self.atom.move_by_displacement(z=PREGRASP_CLEARANCE, xyz_coord='world'),
                  tag='grasp_side_clearance', time_dilation_factor=0.5)
        target = self.cup.get_pose().add_bias([0.0, 0.0, CUP_GRASP_DZ], coord='world')
        gc = construct_grasp_pose(np.array(target.p, dtype=float), gf, camera_up)
        gc_side = gc.add_bias((gf * CUP_SIDE_APPROACH).tolist(), coord='world')
        ee_side = self._robot_manager.gripper_center_to_ee(gc_side)
        ee_high = ee_side.add_bias([0.0, 0.0, CUP_SIDE_UP], coord='world')
        ee_now = self._robot_manager.get_ee_pose()
        ee_transit = Pose([ee_high.p[0], ee_high.p[1], ee_now.p[2]], ee_now.q)
        self.move(self.atom.move_to_pose(ee_transit), tag='grasp_side_horizontal')
        self.move(self.atom.move_to_pose(ee_high), tag='grasp_side_presolve')
        self.move(self.atom.move_to_pose(ee_side), tag='grasp_side_drop')
        self.move(self.atom.move_by_displacement(z=CUP_SIDE_APPROACH + 0.01, xyz_coord='local'),
                  tag='grasp_side_insert', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(self.atom.close_gripper(GRIP_CLOSE))    # 闭到 GRIP_CLOSE(不闭到0): 0 会把杯死压在软 gelpad 上 -> 接触不稳, 杯在空中晃/自转
        if self.no_tactile:
            # 运动测试模式(UNIVTAC_NO_TACTILE=1): 无 gelpad 接触, 靠 weld 把杯刚性绑到夹爪才能搬运/倾倒
            self.weld_actor(self.cup, self._robot_manager)

    def _pour(self):
        """倒球: 直接 move 到盘上方的倾倒预备位, 然后分步翻杯。

        去掉"先抬一段再按位移横挪"这类中间动作, 直接把当前夹爪中心 move_to_pose 到盘上方,
        让路径更短更干净, 也减少抓起后到倾倒前的无效摆动。
        """
        gc0 = self._robot_manager.get_gripper_center_pose()
        gc_prep = Pose([PLATE_POS.p[0] + POUR_PREP_X, PLATE_POS.p[1] + POUR_OVER_Y, POUR_Z], gc0.q)
        ee_prep = self._robot_manager.gripper_center_to_ee(gc_prep)
        self.move(self.atom.move_to_pose(ee_prep), tag='move_over_plate', time_dilation_factor=0.5)

        # 直接在关节空间翻 wrist joint，绕过末端位姿规划失败。
        arm_q = self._robot_manager.robot.data.joint_pos[0, :7].clone()
        start_q7 = arm_q[6].item()
        target_q7 = start_q7 - POUR_ANGLE
        q7_seq = torch.linspace(start_q7, target_q7, POUR_JOINT_STEPS, device=self.device)
        self.atom_id += 1
        self.atom_tag = 'pour'
        for q7 in q7_seq:
            q = arm_q.clone()
            q[6] = q7
            self._robot_manager.set_arm(q, force=True)
            self._step(is_save=True)
        self._update_render()
        self.delay(30, is_save=True)   # 倒后停留, 让球落定在盘上

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(15)            # 让杯/球落稳
        self.move(self.atom.open_gripper(1.0))

    def _play_once(self):
        self._grasp_cup()                                                   # 抓住杯
        self._pour()                                                        # 移到盘上方并倾倒

    # ---------------------------------------------------------------- success
    def check_success(self):
        bp = self.ball.get_pose()
        px, py = PLATE_POS.p[0], PLATE_POS.p[1]
        horiz = float(np.linalg.norm(np.array([bp.p[0] - px, bp.p[1] - py])))
        bz = float(bp.p[2])
        # 成功 = 球落在盘面半径内 且 贴在盘上(低高度) 且 已倒出杯子。
        # ★修复(2026-07-14): 原判据只看球在盘区域+低, 不检查球是否离开杯; 机器人把杯(球还在里)
        #   移到盘上方放低就误判成功+早停("还没倒出来")。加"球离开杯口"要求真倒出。
        cup_p = self.cup.get_pose().p
        ball_to_cup = float(np.linalg.norm(np.array([bp.p[0] - cup_p[0], bp.p[1] - cup_p[1]])))
        on_plate = horiz < PLATE_R - 0.005          # 落在盘内(留 5mm 余量)
        low = bz < 0.03                             # 贴在盘上(没卡在杯里/半空)
        ball_out = ball_to_cup > CUP_TOP_R          # 球水平离开杯口 => 真倒出(非仍在杯内)
        self.metadata['ball_xyz'] = [float(v) for v in bp.p]
        self.metadata['ball_to_plate_horiz'] = horiz
        self.metadata['ball_to_cup_horiz'] = ball_to_cup
        print(f"[POUR] ball=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f}) "
              f"horiz={horiz*1000:.1f}mm to_cup={ball_to_cup*1000:.1f}mm "
              f"on_plate={on_plate} low={low} ball_out={ball_out}", flush=True)
        return bool(on_plate and low and ball_out)
