from tacex_assets.robots.franka.franka_gsmini_gripper_uipc_high_res import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)
from tacex_assets.robots.franka.franka_xensews_gripper_uipc import (
    FRANKA_PANDA_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)
from tacex_assets.robots.franka.franka_gf225_gripper_uipc import (
    FRANKA_PANDA_ARM_GF225_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from ..sensors.tactile import TactileCfg, create_tactile_cfg

@configclass
class RobotCfg:
    robot: ArticulationCfg = None
    tactiles: list[TactileCfg] = []

    gripper_offset: float = 0.131 # in m
    gripper_max_qpos: float = 0.039 # in m

    tactile_far_plane: float = 30.0 # in mm
    adaptive_grasp_depth_threshold: float = 27.5 # in mm, used for grasping
    contact_threshold: tuple[float, float] = (27.5, 28.0) # in mm, used in `gravity_rotate` api

def create_franka_gsmini_gripper(
    data_type:list[str],
    prim_path:str="/World/envs/env_.*/Robot",
    base_pos:tuple[float, float, float]|None=None,
    base_rot:tuple[float, float, float, float]|None=None,
    name_suffix:str="",
):
    """构造 Franka + gsmini 夹爪的 RobotCfg。

    默认 prim 在 /World/envs/env_.*/Robot、基座在 env 原点(单臂任务原样不变)。
    双臂时给不同 prim_path / base_pos / base_rot / name_suffix 即可造第二条臂。
    触觉 prim 由 prim_path 推出, 触觉名加 name_suffix(全局唯一, 否则两臂触觉重名)。
    """
    init_kwargs = dict(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": 0.0,
            "panda_joint3": 0.0,
            "panda_joint4": -2.46,
            "panda_joint5": 0.0,
            "panda_joint6": 2.5,
            "panda_joint7": 0.741,
            "panda_finger.*": 0.02,
        }
    )
    if base_pos is not None:
        init_kwargs["pos"] = base_pos
    if base_rot is not None:
        init_kwargs["rot"] = base_rot
    robot = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(**init_kwargs),
    )
    tactiles = [
        create_tactile_cfg(
            prim_path=f"{prim_path}/gelsight_mini_case_left",
            gelpad_prim_path=f"{prim_path}/gelpad_left",
            gelpad_attachment_body_name="gelsight_mini_case_left",
            name=f"left_tactile{name_suffix}",
            sensor_type="gsmini",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path=f"{prim_path}/gelsight_mini_case_right",
            gelpad_prim_path=f"{prim_path}/gelpad_right",
            gelpad_attachment_body_name="gelsight_mini_case_right",
            name=f"right_tactile{name_suffix}",
            sensor_type="gsmini",
            data_type=data_type,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=34.0,
        adaptive_grasp_depth_threshold=27.5,
        contact_threshold=(27.5, 28.0)
    )


# 双臂(gsmini)布局: 两条 Franka 基座沿 y 分开、均朝 +x(并排, 类 ALOHA), 共享桌前工作区(x≈0.5)。
# 先用并排(无 yaw)避免角度导致两臂/桌面初始穿透; 如需更好可达性再 [tune] 加小 yaw。
# 绕 z 轴 yaw θ 的四元数 = (cos θ/2, 0, 0, sin θ/2)。
DUAL_ARM_A_POS = (0.0, 0.35, 0.0)       # 左臂(举套子)
DUAL_ARM_A_ROT = (1.0, 0.0, 0.0, 0.0)   # 朝 +x
DUAL_ARM_B_POS = (0.0, -0.35, 0.0)      # 右臂(插螺丝)
DUAL_ARM_B_ROT = (1.0, 0.0, 0.0, 0.0)   # 朝 +x

def create_franka_gsmini_gripper_dual(data_type:list[str]):
    """返回 (arm_a_cfg, arm_b_cfg)。arm A 走默认 /Robot prim(沿用现有资产/触觉路径),
    arm B 走 /Robot_b、触觉名带 _b 后缀。两臂基座位姿各自偏移。"""
    arm_a = create_franka_gsmini_gripper(
        data_type, prim_path="/World/envs/env_.*/Robot",
        base_pos=DUAL_ARM_A_POS, base_rot=DUAL_ARM_A_ROT, name_suffix="",
    )
    arm_b = create_franka_gsmini_gripper(
        data_type, prim_path="/World/envs/env_.*/Robot_b",
        base_pos=DUAL_ARM_B_POS, base_rot=DUAL_ARM_B_ROT, name_suffix="_b",
    )
    return arm_a, arm_b

def create_franka_gf225_gripper(data_type:list[str]):
    robot = FRANKA_PANDA_ARM_GF225_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ), 
    )
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/GF225_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/GF225_gelpad_left",
            gelpad_attachment_body_name="GF225_left",
            name="left_tactile",
            sensor_type="gf225",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/GF225_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/GF225_gelpad_right",
            gelpad_attachment_body_name="GF225_right",
            name="right_tactile",
            sensor_type="gf225",
            data_type=data_type,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=26.5,
        adaptive_grasp_depth_threshold=25.3,
        contact_threshold=(25.5, 26.3)
    )

def create_franka_xensews_gripper(data_type:list[str]):
    robot = FRANKA_PANDA_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ),
    )
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/XenseWS_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/XenseWS_gelpad_left",
            gelpad_attachment_body_name="XenseWS_left",
            name="left_tactile",
            sensor_type="xensews",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/XenseWS_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/XenseWS_gelpad_right",
            gelpad_attachment_body_name="XenseWS_right",
            name="right_tactile",
            sensor_type="xensews",
            data_type=data_type,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=30.0,
        adaptive_grasp_depth_threshold=27.3,
        contact_threshold=(27.5, 27.8)
    )