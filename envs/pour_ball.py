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
#   抓取改用 grasp_actor(和 screw 一致, 稳; 之前手写带约束的直线进刀规划失败)。抓后 weld 杯到夹爪。
#   场景中心放在 (0.5, 0)(基座正前方), 与 screw 任务一致。
# ============================================================================

# ---- 几何 (m) ----
CUP_SCALE  = 0.62                    # 缩小到口外⌀~47mm: 远小于夹爪最大开度(~78mm), 从上方下夹时两指有充裕余量,
                                     # 不再像 0.92(口⌀~70mm)那样卡住宽口沿把杯碰倒/抓空
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
PLATE_POS    = Pose([0.50,  0.10, PLATE_CZ], UPRIGHT)         # 盘(接收面/倾倒目标)
CUP_POS      = Pose([0.50, -0.10, TABLE_TOP + CUP_HALF + GAP], UPRIGHT)   # 杯(被抓, 内含球)
BALL_DZ      = -CUP_HALF + 0.006 + BALL_R                     # 球相对杯心的 z(落在杯内底, 不穿壁)

# ---- 抓取 / 倾倒 [tune] ----
CUP_GRASP_DZ = 0.012                 # 抓取点相对杯心的 +z: 取到杯上半身(口沿下方), 从上方下夹这一圈杯壁
LIFT_RISE    = 0.16                  # 抓后竖直举起高度
POUR_OVER_Y  = 0.0                   # 倾倒位杯心 = 盘心正上方(球基本竖直落到盘心)
POUR_Z       = 0.24                  # 倾倒时杯体心世界 z(抬高: 翻杯时 EE 绕夹持点下摆, 夹持点高些 EE 才不会摆到够不到的低位姿)
POUR_ANGLE   = 125.0 / 180.0 * np.pi    # 翻转角: 翻到口朝下 35°(>90°)球已倒出; 停在腕关节极限(~135°)之前避免规划失败
POUR_STEPS   = 12                    # 分步翻转(细 -> 柔, 球不被甩远)


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
        noise = self.create_noise([0.01, 0.01, 0.0])
        cup = CUP_POS.add_offset(noise)
        self.dish.set_pose(PLATE_POS)
        self.cup.set_pose(cup)
        self.ball.set_pose(Pose([cup.p[0], cup.p[1], cup.p[2] + BALL_DZ], UPRIGHT))
        self.metadata['plate_xy'] = [float(PLATE_POS.p[0]), float(PLATE_POS.p[1])]

    # ---------------------------------------------------------------- helpers
    def _grasp_cup(self):
        """从【正上方】抓取杯子(同 screw 任务范式, 用 grasp_actor): 夹爪竖直朝下, 张开跨过杯口, 两指沿杯外壁
           下落到杯上部(口沿下一点), 自适应闭合从两侧(±x)夹住这圈杯壁。
           grasp_actor 自动: 先到抓取点正上方 pre_dis 处 -> 竖直下降到抓取点 -> (此处不闭合, 之后单独自适应闭合)。
           CUP.usd 无预设抓取点, 用 construct_grasp_pose 现算并登记一个 contact 点。"""
        grasp_from = np.array([0.0, 0.0, 1.0])   # 从正上方接近(夹爪 z 轴朝下)
        camera_up = np.array([0.0, -1.0, 0.0])   # 同 screw 套筒 top-down: 指头沿 world-x 开合 -> 夹住杯 ±x 壁
        self.move(self.atom.open_gripper(1.0))
        target = self.cup.get_pose().add_bias([0.0, 0.0, CUP_GRASP_DZ], coord='world')   # 杯上部高度
        gc = construct_grasp_pose(np.array(target.p, dtype=float), grasp_from, camera_up)
        self.cup_grasp_id = self.cup.register_point(gc, type='contact')     # 给杯登记 contact 抓取点(资产本身无预设点)
        self.move(self.atom.grasp_actor(self.cup, contact_point_id=self.cup_grasp_id,
                                        pre_dis=0.12, dis=0.0, is_close=False))   # 从上方下降到抓取点
        self.move(self.atom.close_gripper(0.0))    # 直接 100% 闭合夹死(非自适应); 物理上指头停在杯壁, 持续夹紧不打滑

    def _pour(self):
        """倒球: ① 把杯(连同在手)平移到盘上方(保持竖直) -> ② 用引擎自带 gripper_rotate 翻杯倒出
           (lift_can 倒罐同款原语: 绕杯自身轴逐步翻、保持夹爪朝向, curobo 可解、不甩到够不到的位姿)。"""
        # ① 平移到盘上方(杯保持竖直)。用【相对位移】而非在手绝对位姿: 不 weld 时杯会在指间微滑,
        #    用 inhand 反算的绝对夹爪目标会漂到够不到处(规划失败)。改为按"杯当前实际位置 -> 目标"的世界位移
        #    平移夹爪, 杯随夹爪一起走, 鲁棒且不依赖在手变换。
        cup_now = np.array(self.cup.get_pose().p, dtype=float)
        target_xyz = np.array([PLATE_POS.p[0], PLATE_POS.p[1] + POUR_OVER_Y, POUR_Z], dtype=float)
        d = target_xyz - cup_now
        self.move(self.atom.move_by_displacement(x=float(d[0]), y=float(d[1]), z=float(d[2]), xyz_coord='world'),
                  tag='move_over_plate', time_dilation_factor=0.5)
        # ② 翻杯倒出: 直接【设新的夹爪位姿】来转手腕(gripper_rotate 只挪位置不转朝向, 手腕几乎不动)。
        #    取当前夹爪中心位姿, 绕【过夹持点的 world-x 轴】把朝向旋转 POUR_ANGLE(位置保持在盘上方),
        #    分步插值 move_to_pose -> 杯子真正翻过来(口朝下), 球倒到盘上。
        gc0 = self._robot_manager.get_gripper_center_pose()
        for i in range(1, POUR_STEPS + 1):
            ang = POUR_ANGLE * i / POUR_STEPS
            gc_i = gc0.add_rotation([ang, 0.0, 0.0], coord='local')   # 绕 world-x 翻朝向, 夹持点位置不动
            ee_i = self._robot_manager.gripper_center_to_ee(gc_i)
            self.move(self.atom.move_to_pose(ee_i), tag='pour', time_dilation_factor=0.5)
        self.delay(30, is_save=True)   # 倒后停留, 让球落定在盘上

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(15)            # 让杯/球落稳
        self.move(self.atom.open_gripper(1.0))

    def _play_once(self):
        self._grasp_cup()                                                   # 抓住杯
        self.move(self.atom.move_by_displacement(z=LIFT_RISE, xyz_coord='world'),
                  tag='lift', time_dilation_factor=0.5)                     # 竖直举起(不加约束, 侧抓后局部 z 是水平的)
        self._pour()                                                        # 移到盘上方并倾倒

    # ---------------------------------------------------------------- success
    def check_success(self):
        bp = self.ball.get_pose()
        px, py = PLATE_POS.p[0], PLATE_POS.p[1]
        horiz = float(np.linalg.norm(np.array([bp.p[0] - px, bp.p[1] - py])))
        bz = float(bp.p[2])
        # 成功 = 球落在盘面半径内 且 贴在盘上(低高度)。
        on_plate = horiz < PLATE_R - 0.005          # 落在盘内(留 5mm 余量)
        low = bz < 0.03                             # 贴在盘上(没卡在杯里/半空)
        self.metadata['ball_xyz'] = [float(v) for v in bp.p]
        self.metadata['ball_to_plate_horiz'] = horiz
        print(f"[POUR] ball=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f}) "
              f"horiz={horiz*1000:.1f}mm on_plate={on_plate} low={low}", flush=True)
        return bool(on_plate and low)
