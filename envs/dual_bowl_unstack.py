from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Dual Bowl UN-STACK —— 拆叠碗: 开局两只碗已套叠(上碗坐在下碗里), 机器人抓住上碗【碗沿/边缘】
#   竖直抽出、移到一侧放下, 把两只碗分开。结构仿照 dual_cup_stack(拆叠纸杯), 区别:
#     1) 物体换成 BOWL.usd(⌀130mm 浅口陶碗, 5mm 壁, 带圈足; 碗口朝 local +Z, 闭底 -Z, 体心居中)
#     2) 抓【碗沿】: 碗口⌀130mm > 夹爪最大开度, 无法像纸杯那样整只跨抱 -> 必须只夹住碗沿一侧
#        (一指落入碗内壁、一指在碗外壁, 沿径向钳夹这一小段约 5mm 厚的碗壁)。
#     3) 不用自适应抓取(adaptive grasp): use_adaptive_grasp=False + close_gripper(0.0) 直接 100% 闭合到底。
#   下碗靠自重(高密度+高桌面摩擦)留在桌上不被带起; 套叠碗壁摩擦"焊死"问题同纸杯, 降 μ 到 5 化解。
# ----------------------------------------------------------------------------
#   BOWL 局部(原点居中, z∈[-half,+half]): 碗口 local +Z, 闭底 local -Z; 原点->口沿/底 = BOWL_HALF。
# ============================================================================

# ---- 几何 (m) ----
BOWL_HALF    = 0.0197    # 碗半高(原点->口沿 / 原点->底); USD 已是 ⌀130mm/高39.4mm 实尺, 故场景 scale=1.0
BOWL_OUTER_R = 0.065     # 碗口外半径(⌀130mm, 仅口沿最顶端那一圈)
# 注意: 碗向下迅速收口。口沿顶端壁很薄(0.5-1mm, r≈0.064-0.065), 越往下壁越厚:
#   口下 3mm: 壁 r∈[0.0588,0.0633](≈4.5mm 厚); 口下 5mm: r∈[0.0584,0.0618](≈3.4mm)。
# 抓在【口下约 5mm】处(壁实、且对落点高度误差宽容), 夹爪中心取该处壁厚中线半径 ≈0.060:
# 此半径在口下 2-8mm 区间都落在壁内 -> 即使下降落点差几 mm, 碗壁仍夹在两指之间(不漏抓)。
RIM_GRASP_R  = 0.060     # 夹爪中心(TCP)到碗轴的水平距离(落在口下 5mm 处壁厚中线)
FOOT_R       = 0.026     # 圈足外半径
TABLE_TOP    = 0.004     # 桌面顶高(碗底贴此面); 碗体心 = TABLE_TOP + BOWL_HALF
PLATE_HALF_Z = 0.004     # 目标盘半高(PLATE.usd, 作为可视化落点提示)
PLATE_CZ     = TABLE_TOP + PLATE_HALF_Z
PLATE_R      = 0.10      # 目标盘外半径(Ø200)

# 套叠: 新 bowl 资产把内底/圈足抬高了, 套叠时上碗会自然坐得更高。
# 相应提高初始上碗体心抬高量, 让任务摆位与新几何一致、并给抓沿更多可见空间。
NEST_RISE    = 0.038
NEST_GAP     = 0.014     # 再把初始套叠间隙拉大一些, 让两碗更容易分离

# ---- 摆位 [tune] (世界系) ----
UPRIGHT      = [1, 0, 0, 0]
STACK_POS    = Pose([0.45, 0.00, TABLE_TOP + BOWL_HALF], UPRIGHT)   # 下碗体心(贴桌, 离臂近些便于够低位)
# 抽出的上碗放到 +Y 一侧桌面(与下碗水平分开 -> 判拆开成功)。
PLATE_POSE   = Pose([0.45, 0.30, PLATE_CZ], UPRIGHT)
PLACE_POSE   = Pose([PLATE_POSE.p[0], PLATE_POSE.p[1], TABLE_TOP + 2 * PLATE_HALF_Z + BOWL_HALF], UPRIGHT)
PLACE_SETTLE_DROP = 0.008

# ---- 抓取 [tune] ----
# 上碗【碗沿】抓取: 从正上方下到上碗 +Y 侧口沿(离 A 臂最近的一段沿, 最好够), 径向钳夹该壁。
# 注意 TCP(夹爪中心)在【胶垫接触面下方】约 1cm: 把 TCP 放到口沿处时, 胶垫还悬在碗口上方没碰到碗
# (实测没夹到)。新 bowl 资产碗沿更外露, 抓取点再上移一些, 更靠近可夹的上缘实壁。
RIM_GRASP_DZ = BOWL_HALF - 0.006
EXTRACT_RISE = 0.16                # 竖直抽出上碗的高度(再抬高些, 先明显脱离堆叠后再侧向搬运)
RIM_DOWN_EXTRA = 0.020             # 竖直下探时再多压 20mm, 让胶垫更实地落到碗沿/内外壁上
RELEASE_HOVER_Z = 0.16             # 放置区上方的明确悬停高度, 先到盘正上方再下放


@configclass
class TaskCfg(BaseTaskCfg):
    dual_arm = True       # <- 开启双臂(自动建 arm B: /Robot_b + 触觉 *_b); 本任务仅用 A 臂(同纸杯拆叠)
    use_adaptive_grasp = False   # <- 关掉自适应抓取: 抓碗沿改为直接 100% 闭合(close_gripper(0.0))夹到底
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
    step_lim = 1200
    reset_time_limit = 1200.0   # 双臂 + 两个套叠 UIPC 软体碗接触, reset 慢, 给足余量防超时


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        # 之前高摩擦虽然更利于夹住碗沿, 但会显著加重套叠碗之间的接触求解负担, 仿真变慢。
        # 这里降到中高摩擦, 在保持一定夹持力的同时减轻嵌套接触的计算压力。
        cfg.sim.physics_material.dynamic_friction = 5.0
        cfg.sim.physics_material.static_friction = 5.0
        cfg.uipc_sim.contact.default_friction_ratio = 5.0
        # 两碗都是被操作件且开局即套叠(原地不动)。规划器冻结的碰撞世界在 super().__init__() 内创建,
        # 那时若没忽略, 套叠摞正好在抓取位 -> action0 必撞 -> 规划失败。必须在 super().__init__() 之前
        # 设任务级 ignore(get_curr_world_cfg 读 task 级∪各臂 owning_manager 级)。
        self.planner_ignore_actors = {'bowl_a', 'bowl_b', 'dish'}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {'bowl_a', 'bowl_b', 'dish'}      # A 抓上碗, 要穿过/脱离下碗
        self._robot_manager_b.ignore_actors = {'bowl_a', 'bowl_b', 'dish'}    # B 闲置, 一并忽略避免冻结世界误撞

    # NOTE: 不重写 get_frame_shot —— 基类已按双臂自动拼出 1760x320 帧
    # (head|wrist|wrist_b 各 480 + 4 路触觉 2 列×160), 正好等于 video_size, 不会被拉伸。

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        self.dish = self._actor_manager.add_from_usd_file(
            name='dish', asset_path="PLATE.usd", pose=PLATE_POSE, density=1e4,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(kinematic=True))
        # 两只相同陶碗(同一 USD, 不同实例名); 初始位姿在 _reset_actors 里摆成套叠摞。
        # 软体碗默认很轻, 闭爪/抽出时会被推跑。加大密度(配高桌面摩擦)让下碗压得住、留在桌上;
        # 2e5 -> 每碗约 11kg*体积分数 ... 实际约 0.9-1kg, 仍在 Franka 负载内, 单边抽出无碍。
        # 下碗很重(2e5 -> ~11kg)+ 高摩擦, 抽上碗时稳稳留在桌上不被带起。
        # 上碗轻(5e3 -> ~0.28kg): 单边碗沿抓取力臂大(碗心离夹持点 ~6cm), 轻碗力矩小 ->
        # 夹持(线接触)能扛住、抽出时不打滑/不甩翻; 太重则边抓边抽必滑脱。
        self.bowl_a = self._actor_manager.add_from_usd_file(
            name='bowl_a', asset_path="BOWL.usd", pose=STACK_POS, density=2e5)
        self.bowl_b = self._actor_manager.add_from_usd_file(
            name='bowl_b', asset_path="BOWL.usd", density=3e3,
            pose=Pose([STACK_POS.p[0], STACK_POS.p[1], STACK_POS.p[2] + NEST_RISE + NEST_GAP], UPRIGHT))

    def _reset_actors(self):
        # 开局即套叠: 下碗在桌, 上碗正立坐在下碗里(体心高 NEST_RISE, 同 xy + 小噪声)。
        noise = self.create_noise([0.002, 0.002, 0.0])   # 小噪声: 抓碗沿本就吃紧, 减随机提高可重复
        base = STACK_POS.add_offset(noise)
        self.dish.set_pose(PLATE_POSE)
        self.bowl_a.set_pose(base)
        self.bowl_b.set_pose(Pose([base.p[0], base.p[1], base.p[2] + NEST_RISE + NEST_GAP], UPRIGHT))
        self.metadata['plate_xy'] = [float(PLATE_POSE.p[0]), float(PLATE_POSE.p[1])]

    # ---------------------------------------------------------------- helpers
    def _grasp_rim(self, actor, rm, atom, arm, rim_dir=(0, 1, 0), camera_up=(1, 0, 0),
                   dz=RIM_GRASP_DZ, side=0.05, up=0.06):
        """从正上方抓住碗的【一侧碗沿】(径向钳夹这一段碗壁):
           rim_dir = 选取碗沿的水平方向(默认 +Y, 离 A 臂最近一侧, 最好够); 夹爪中心落在该侧沿壁中线。
           camera_up 选得使夹爪指开合方向 = 径向(rim_dir): 一指落碗内壁、一指在碗外壁。
           两段式接近: 沿沿点正上方悬停 -> 竖直降到沿上 -> 沿夹爪局部 z(竖直)进刀贴住沿 -> 100% 闭合。"""
        rd = np.array(rim_dir, dtype=float); rd = rd / np.linalg.norm(rd)
        self.move(atom.open_gripper(1.0), arm=arm)
        center = actor.get_pose()
        # 碗沿抓取点 = 碗体心 + 径向 RIM_GRASP_R(到口下 5mm 处壁厚中线) + 竖直 dz(口沿顶下方 5mm)
        gp = np.array([center.p[0], center.p[1], center.p[2]], dtype=float) + rd * RIM_GRASP_R + np.array([0, 0, dz])
        gc = construct_grasp_pose(gp, np.array([0, 0, 1], dtype=float), np.array(camera_up, dtype=float))
        gc_high = gc.add_bias([0.0, 0.0, side + up], coord='world')    # 沿点正上方(悬停)
        ee_high = rm.gripper_center_to_ee(gc_high)
        self.move(atom.move_to_pose(ee_high), arm=arm)                 # 1) 到沿正上方悬停
        # 2) 像 HDMI 一样把最后的下探拆成两小段受约束直线进刀, 减少长距离单次规划的不稳定。
        down_total = side + up + RIM_DOWN_EXTRA
        self.move(atom.move_by_displacement(z=down_total * 0.75, xyz_coord='local'), arm=arm,
                  constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(atom.move_by_displacement(z=down_total * 0.25, xyz_coord='local'), arm=arm,
                  constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        tcp = rm.get_gripper_center_pose()
        msg = (f"[DBG] grasp-target gp=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f})  "
               f"TCP-actual=({tcp.p[0]:.3f},{tcp.p[1]:.3f},{tcp.p[2]:.3f})  rim_top_z={center.p[2]+BOWL_HALF:.4f}")
        try:  # 打印手指/胶垫体的世界 z, 用来标定 "胶垫接触面 vs TCP" 的竖直偏移
            ids, names = rm.robot.find_bodies('.*finger.*')
            fz = rm.robot.data.body_link_pos_w[0, ids]
            msg += "  fingers[" + " ".join(f"{names[i]}z={fz[i,2].item():.4f}" for i in range(len(names))) + "]"
        except Exception as e:
            msg += f"  (finger-pos err: {e})"
        print(msg, flush=True)
        # 3) 直接闭满, 让两指尽可能压紧碗沿。
        self.move(atom.close_gripper(0.0), arm=arm)
        self._close_gripper_direct(rm, 0.0, settle_steps=10, is_save=True)

    def _move_gripper_direct(self, rm, atom, target_pose, arm):
        """保持当前夹爪朝向不变, 直接把夹爪中心移动到目标世界位置。"""
        gc_now = rm.get_gripper_center_pose()
        target_gc = Pose(list(target_pose.p), gc_now.q)
        target_ee = rm.gripper_center_to_ee(target_gc)
        self.move(atom.move_to_pose(target_ee), arm=arm, time_dilation_factor=0.5)

    def _open_gripper_direct(self, rm, percent=1.0, settle_steps=12, is_save=True):
        target = rm.gripper_percent2qpos(percent)
        pos = torch.tensor([target, target], device=rm.device, dtype=rm.robot.data.joint_pos.dtype)
        self.atom_id += 1
        self.atom_tag = 'open_direct'
        rm.set_gripper(pos, force=True)
        for _ in range(settle_steps):
            rm.set_gripper(pos, force=True)
            self._step(is_save=is_save)
        self._update_render()

    def _close_gripper_direct(self, rm, percent=0.0, settle_steps=10, is_save=True):
        target = rm.gripper_percent2qpos(percent)
        pos = torch.tensor([target, target], device=rm.device, dtype=rm.robot.data.joint_pos.dtype)
        self.atom_id += 1
        self.atom_tag = 'close_direct'
        rm.set_gripper(pos, force=True)
        for _ in range(settle_steps):
            rm.set_gripper(pos, force=True)
            self._step(is_save=is_save)
        self._update_render()

    def _dbg(self, tag):
        ap, bp = self.bowl_a.get_pose(), self.bowl_b.get_pose()
        ga = self._robot_manager.get_gripper_qpos()
        print(f"[DBG] {tag}: gripA={ga:.4f}  "
              f"bowl_a=({ap.p[0]:.3f},{ap.p[1]:.3f},{ap.p[2]:.3f})  "
              f"bowl_b=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f})", flush=True)

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(20)   # 让套叠摞落稳; 抓取/拆叠全放到 _play_once(会被录进视频)
        self._dbg("套叠落稳")

    def _play_once(self):
        # A 臂从正上方抓住上碗 +Y 侧【碗沿】, 竖直抽出脱离下碗, 移到 +Y 侧桌面放下、松爪。
        # 下碗(2e5 重密度 + 高摩擦)靠自重稳稳留在桌上, 无需另一臂夹持。
        self._grasp_rim(self.bowl_b, self._robot_manager, self.atom_a, 'a',
                        rim_dir=(0, 1, 0), camera_up=(1, 0, 0))
        self._dbg("A 夹住上碗碗沿")

        # A: 竖直向上抽出上碗(沿世界 +Z), constraint_pose 锁其余 5 轴 -> 直线竖直脱离下碗。
        self.move(self.atom_a.move_by_displacement(z=EXTRACT_RISE * 0.7, xyz_coord='world'),
                  arm='a', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(self.atom_a.move_by_displacement(z=EXTRACT_RISE * 0.3, xyz_coord='world'),
                  arm='a', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self._dbg("A 抽出上碗(脱离下碗)")

        # A: 把上碗移到 +Y 侧桌面上方 -> 下放贴桌 -> 松爪释放。
        hover = Pose([PLATE_POSE.p[0], PLATE_POSE.p[1], PLACE_POSE.p[2] + RELEASE_HOVER_Z], PLACE_POSE.q)
        self._move_gripper_direct(self._robot_manager, self.atom_a, hover, 'a')
        self._dbg("A 移到放置点上方")
        # 从盘正上方只沿世界 -Z 下放, 避免最后一个落点仍偏离盘中心。
        gripper_now = self._robot_manager.get_gripper_center_pose()
        drop_z = (PLACE_POSE.p[2] - gripper_now.p[2])   # 修:下降到位(原0.5*只降一半→碗在盘上方~66mm就松爪没落盘)
        self.move(self.atom_a.move_by_displacement(z=float(drop_z), xyz_coord='world'),
                  arm='a', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(self.atom_a.move_by_displacement(z=-PLACE_SETTLE_DROP, xyz_coord='world'),
                  arm='a', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.delay(8, is_save=True)                         # 先让碗在桌上落稳, 再松爪
        self._open_gripper_direct(self._robot_manager, 1.0, settle_steps=14, is_save=True)   # 松爪释放上碗
        self._dbg("A 放下上碗并松爪")
        self.delay(25, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        ap = self.bowl_a.get_pose()
        bp = self.bowl_b.get_pose()
        # 拆开成功 = 两碗已分离 + 上碗正立 + 上碗落在目标盘范围内。
        horiz = float(np.linalg.norm(np.array(ap.p[:2], dtype=float) - np.array(bp.p[:2], dtype=float)))
        separated = horiz > 0.12
        a_up = float(np.dot(ap.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.7
        b_up = float(np.dot(bp.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.6
        plate_horiz = float(np.linalg.norm(np.array(bp.p[:2], dtype=float) - np.array(PLATE_POSE.p[:2], dtype=float)))
        on_plate = plate_horiz < (PLATE_R - BOWL_OUTER_R + 0.01)
        self.metadata['bowl_a'] = [float(v) for v in ap.p]
        self.metadata['bowl_b'] = [float(v) for v in bp.p]
        self.metadata['horiz_sep'] = horiz
        self.metadata['bowl_b_to_plate_horiz'] = plate_horiz
        print(f"[UNSTACK] horiz_sep={horiz*1000:.1f}mm plate_err={plate_horiz*1000:.1f}mm "
              f"a_up={a_up} b_up={b_up} on_plate={on_plate} -> separated={separated}", flush=True)
        pa = self._robot_manager.get_gripper_percentage()
        released = pa > 0.9   # A gripper opened = bowl actually let go
        print(f"[UNSTACK] gripperA={pa:.2f} released={released}", flush=True)
        return bool(separated and b_up and on_plate and released)
