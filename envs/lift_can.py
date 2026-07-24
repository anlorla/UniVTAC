from ._base_task import *
import numpy as np
import os

WRIST_TILT_DEG = -30.0  # pitch the wrist cam (about its optical X) down toward the gripper
WRIST_FOCAL = 1.3       # smaller focal length than baseline (1.94) -> wider FOV (None = leave)

# Fixed-cam variant switch. The wrist-cam tilt+wide-FOV adjustment below was added
# 2026-07-04 (b375934) and until now ran UNCONDITIONALLY, so every collect/eval since
# then used the "fixed cam" view. Data collected BEFORE that change used the ORIGINAL
# wrist camera (no tilt, focal 1.94). Train/eval MUST use the SAME camera, so the
# fixed-cam adjustment is now OPT-IN and kept distinct from the original:
#   UNIVTAC_LIFT_CAN_FIXEDCAM=1  -> fixed cam (tilt -30° + focal 1.3)
#   unset / 0 (default)          -> original wrist camera (pre-2026-07-04 behaviour)
LIFT_CAN_FIXEDCAM = os.environ.get('UNIVTAC_LIFT_CAN_FIXEDCAM', '0') == '1'


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 600
    adaptive_grasp_depth_threshold = 27.75


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)
        # OPT-IN: only apply the fixed-cam (tilt+wide) adjustment when explicitly
        # enabled, so it stays distinct from the original wrist camera (see top-of-file).
        if LIFT_CAN_FIXEDCAM:
            self._adjust_wrist_cam()

    def _adjust_wrist_cam(self):
        # Tilt the wrist camera mount down toward the gripper (delta on the working pose)
        # and optionally widen its FOV by lowering focal length. Applied to every env.
        try:
            from pxr import UsdGeom
            import isaacsim.core.utils.stage as stage_utils
            import math
            from pxr import Gf
            stage = stage_utils.get_current_stage()
            for prim in stage.Traverse():
                path = prim.GetPath().pathString
                if path.endswith('/Robot/WristCamera/Camera'):
                    # tilt: post-multiply the camera orient by a pitch about its optical X
                    if WRIST_TILT_DEG:
                        xf = UsdGeom.Xformable(prim)
                        for op in xf.GetOrderedXformOps():
                            if op.GetOpName() == 'xformOp:orient':
                                q = op.Get()  # Gf.Quatd/Quatf (w + imaginary vec)
                                w0, im = q.GetReal(), q.GetImaginary()
                                q0 = Gf.Quatd(float(w0), float(im[0]), float(im[1]), float(im[2]))
                                a = math.radians(WRIST_TILT_DEG) / 2.0
                                qp = Gf.Quatd(math.cos(a), math.sin(a), 0.0, 0.0)  # about local X
                                qn = (q0 * qp).GetNormalized()
                                op.Set(type(q)(qn.GetReal(), *[float(v) for v in qn.GetImaginary()]))
                    if WRIST_FOCAL is not None:
                        UsdGeom.Camera(prim).GetFocalLengthAttr().Set(float(WRIST_FOCAL))
        except Exception as e:
            print(f"[CAMDBG] err: {e}", flush=True)
 
    def create_actors(self):
        self.cans:dict[int, Actor] = {}
        pose_dict = {
            4: Pose([-1.0, 0.0, 0.022], [1, 0, 0, 0]),
            5: Pose([-1.0, 1.0, 0.027], [1, 0, 0, 0]),
            6: Pose([-1.0, -1.0, 0.032], [1, 0, 0, 0])
        }
        for d in [4, 5, 6]:
            self.cans[d] = self._actor_manager.add_from_usd_file(
                name=f'can_d{d}',
                asset_path=f"Can_d{d}cm.usd",
                pose=pose_dict[d]
            )

    def _reset_actors(self):
        can_offset = self.create_noise([0.02, 0.05, 0.0])
        can_size = self.rng.choice([4, 5, 6])
        can_pose = Pose(
            [0.7, 0.0, 0.005*can_size+0.001], [1, 0, 0, 0]
        ).add_offset(can_offset)
 
        self.can = self.cans[can_size]
        self.metadata['can_size'] = int(can_size)
        self.can.set_pose(can_pose)
 
    def pre_move(self):
        self.delay(10)

        self.move(self.atom.open_gripper(1.0))
        can_pose = self.can.get_pose()
        target_pose = can_pose.add_bias([-0.065, 0, -0.008])
        target_mat = target_pose.to_transformation_matrix()
        x = target_mat[:3, 0].reshape(-1)
        target_mat = np.vstack([
            x, np.cross(x, [0, 0, 1]), [0, 0, 1],
        ])
        self.grasp_noise = self.create_noise(euler=[0, [-np.pi/6, -np.pi/18], 0])
        self.metadata['grasp_noise'] = self.grasp_noise.tolist()
        target_pose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        ).add_offset(self.grasp_noise)
        grasp_idx = self.can.register_point(
            pose=target_pose,
            type='contact'
        )
        self.move(self.atom.grasp_actor(self.can, contact_point_id=grasp_idx, is_close=False))
        self.origin_inhand_pose = self._robot_manager.get_inhand_pose(self.can)
        
    def _play_once(self):
        self.move(self.atom.close_gripper())
        self.gripper_rotate(self.can, 70/180*np.pi, steps=4)
        if not self.check_mid_success():
            self.gravity_rotate(self.can, [0, 0, 1], [-1, 0, 0])
        self.move(self.atom.open_gripper())
        self.delay(30, is_save=False)

    def check_mid_success(self):
        can_pose = self.can.get_pose()
        return np.abs(np.dot(can_pose.to_transformation_matrix()[:3, 0], np.array([0, 0, 1]))) > 0.95

    def check_early_stop(self):
        can_pose = self.can.get_pose()
        inhand_pose = self._robot_manager.get_inhand_pose(self.can)
        min_depth = torch.min(self._tactile_manager.get_min_depth()).item()
        
        if min_depth < 20:
            self.metadata['early_stop'] = True
            self.metadata['min_depth'] = float(min_depth)
            return True
        if np.abs(inhand_pose.p[2] - self.origin_inhand_pose.p[2]) > 0.05 and \
            np.abs(np.dot(can_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.99:
            self.metadata['early_stop'] = True
            self.metadata['inhand_dis'] = float(np.abs(inhand_pose.p[2] - self.origin_inhand_pose.p[2]))
            return True
        return False
 
    def check_success(self):
        can_pose = self.can.get_pose()
        return can_pose[2] < 0.01 and \
            np.abs(np.dot(can_pose.to_transformation_matrix()[:3, 0], np.array([0, 0, 1]))) > 0.99
