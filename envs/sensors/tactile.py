import torch
from envs.utils import data
import numpy as np
from tacex import GelSightSensor, GelSightSensorCfg
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.sensors.gf225.gf225_cfg import GF225Cfg
from tacex.simulation_approaches.fem_based import ManiSkillSimulatorCfg
from tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg

from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg

from tacex_uipc import (
    UipcRLEnv,
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcSimCfg
)

from ..utils.transforms import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._base_task import BaseTask
    from tacex_uipc.sim import UipcIsaacAttachmentsCfg, UipcSim
    from tacex_uipc import UipcInteractiveScene

@configclass
class TactileCfg:
    name: str = 'tactile_sensor'
    sensor_cfg = None
    gelpad_cfg: UipcObjectCfg = None
    gelpad_attachment_cfg: UipcIsaacAttachmentsCfg = None

def create_gelsight_mini_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    name: str = "tactile_sensor",
    resolution = (320, 240),
    update_period = 1/120,
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
):
    from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
    sensor_cfg = GelSightMiniCfg(
        prim_path=prim_path,
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            resolution=resolution,
            update_period=update_period,
            data_types=["depth", "rgb"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=False,  # for rendering sensor output in the gui
        update_period=1/120,
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            tactile_img_res=resolution,
            marker_shape=(9, 7),
            marker_interval=(2.40625, 2.45833),
            sub_marker_num=0,
            marker_radius=6,
            camera_to_surface=0.0283,
            real_size=(0.0266, 0.0209),
            sensor_type='gsmini',
        ),
        data_types=data_type
    )
    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 64
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        with_shadow=False,
        tactile_img_res=resolution,
        device="cuda",
    )

    cfg = TactileCfg(
        name=name,
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcObjectCfg(
            prim_path=gelpad_prim_path,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e4
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=1e4,
            body_name=gelpad_attachment_body_name,
            debug_vis=False,
        ),
    )
    return cfg

def create_gf225_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    data_type: list[str] = ["camera_depth", "tactile_rgb"],
) -> TactileCfg:
    resolution = (480, 480)  # GF225 resolution
    update_period = 1/120
    
    sensor_cfg = GF225Cfg(
        prim_path=prim_path,
        sensor_camera_cfg=GF225Cfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            resolution=resolution,
            update_period=update_period,
            data_types=["depth"],
            clipping_range=(0.02, 0.0265),
        ),
        device="cuda",
        debug_vis=False,
        update_period=1/120,
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            tactile_img_res=resolution,
            sub_marker_num=0,
            marker_radius=8,
            marker_shape=(9, 9),
            marker_interval=(2.0, 2.0),
            camera_to_surface=0.0265,
            real_size = (0.0235, 0.0250),
            sensor_type='gf225',
        ),
        data_types=data_type
    )
    
    from tacex.simulation_approaches.mlp_fots import MLPFOTSSimulatorCfg
    from tacex_assets import TACEX_ASSETS_DATA_DIR

    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 81
    sensor_cfg.optical_sim_cfg = MLPFOTSSimulatorCfg(
        calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/GF225/calibs/480x480",
        tactile_img_res=resolution,
        device="cuda",
    )
    
    cfg = TactileCfg(
        name=name,
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcObjectCfg(
            prim_path=gelpad_prim_path,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e4
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=1e4,
            body_name=gelpad_attachment_body_name,
            isaac_rigid_prim_path=gelpad_attachment_prim_path,
            debug_vis=False,
        ),
    )
    return cfg

def create_xensews_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    resolution = (320, 240),
    update_period = 1/120,
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
) -> TactileCfg:
    from tacex_assets.sensors.xensews.xensews_cfg import XenseWSCfg

    sensor_cfg = XenseWSCfg(
        prim_path=prim_path,
        sensor_camera_cfg=XenseWSCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=update_period,
            resolution=resolution,
            data_types=["depth", "rgb"],
            clipping_range=(0.01, 0.03),  # (0.024, 0.034),
        ),
        device="cuda",
        debug_vis=False,  # for rendering sensor output in the gui
        update_period=update_period,
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            tactile_img_res=resolution,
            sub_marker_num=0,
            sensor_type='xensews',
        ),
        data_types=data_type
    )
    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 1200
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        with_shadow=False,
        tactile_img_res=resolution,
        device="cuda",
    )

    cfg = TactileCfg(
        name=name,
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcObjectCfg(
            prim_path=gelpad_prim_path,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e4
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=1e4,
            body_name=gelpad_attachment_body_name,
            isaac_rigid_prim_path=gelpad_attachment_prim_path,
            debug_vis=False,
        ),
    )
    return cfg

def create_tactile_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    sensor_type:Literal['gsmini', 'xensews', 'gf225'] = "gsmini",
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
) -> TactileCfg:
    if sensor_type == "gsmini":
        return create_gelsight_mini_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            name=name,
            data_type=data_type,
        )
    elif sensor_type == "xensews":
        return create_xensews_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            name=name,
            data_type=data_type,
        )
    elif sensor_type == "gf225":
        return create_gf225_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            gelpad_attachment_prim_path=gelpad_attachment_prim_path,
            name=name,
            data_type=data_type,
        )
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")


class VisualTactileSensor:
    def __init__(self, name:str, cfg:TactileCfg, robot, scene: 'UipcInteractiveScene', uipc_sim:'UipcSim'):
        self.cfg = cfg
        self.name = name
        self.scene = scene
        self.robot = robot
        self.uipc_sim = uipc_sim

        self.gelpad = UipcObject(self.cfg.gelpad_cfg, self.uipc_sim)
        self.attachment = UipcIsaacAttachments(
            self.cfg.gelpad_attachment_cfg, self.gelpad, self.robot
        )
        self.sensor = GelSightSensor(self.cfg.sensor_cfg, self.gelpad)
        # self.scene.sensors[f'tactile_{self.cfg.name}'] = self.sensor
    
    def setup(self):
        self.device = self.uipc_sim.cfg.device
        init_pts = self.gelpad._data.nodal_pos_w[self.attachment.attachment_points_idx].cpu().numpy()
        init_world_trans = self.gelpad.init_world_transform.cpu().numpy()
        self.origin_pts = (init_pts - init_world_trans[:3, 3]) @ (init_world_trans[:3, :3].T).T
        attach_pts = self.attachment.attachment_offsets
        init_trans = estimate_rigid_transform(self.origin_pts, attach_pts)
        self.attach_to_init = np.linalg.inv(init_trans)
        self.attach_to_init = torch.tensor(self.attach_to_init, dtype=torch.float64, device=self.device)

        self.sensor.marker_motion_simulator.marker_motion_sim.init_vertices()

    def get_attach_pose(self):
        if type(self.attachment.isaaclab_rigid_object) is Articulation:
            # this only works when rigid body is an articulation
            # self.attachment.isaaclab_rigid_object._physics_sim_view.update_articulations_kinematic()
            # read data from simulation
            poses = self.attachment.isaaclab_rigid_object._root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            pose = poses[:, self.attachment.rigid_body_id, 0:7].clone()
        elif type(self.attachment.isaaclab_rigid_object) is RigidObject:
            # only works with rigid body
            pose = self.attachment.isaaclab_rigid_object._root_physx_view.root_state_w.view(-1, 1, 13)
            pose = pose[:, self.attachment.rigid_body_id, 0:7].clone()
        else:
            raise RuntimeError("Need an Articulation or a RigidBody object for the Isaac X UIPC attachment.")
        return Pose.from_list(pose.flatten().tolist())

    def get_init_pts(self):
        curr_attach_pose = self.get_attach_pose()
        trans_to_attach = np.linalg.inv(curr_attach_pose.to_transformation_matrix())
        trans_to_attach = torch.tensor(trans_to_attach, dtype=torch.float64, device=self.device)
        trans_to_init = self.attach_to_init @ trans_to_attach
        return self.gelpad.data.nodal_pos_w @ trans_to_init[:3, :3].T + trans_to_init[:3, 3]
 
    def update(self, dt, force_recompute=False):
        self.gelpad.update(dt=dt)
        self.sensor.update(dt=dt, force_recompute=force_recompute)
    
    def set_debug_vis(self):
        if not self.sensor.cfg.debug_vis:
            return 
        for data_type in ['marker_motion']:
            self.sensor._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
    
    def get_observations(self, data_types: list[str] = None):
        obs = {}
        if data_types is None:
            data_types = ['rgb', 'rgb_marker', 'depth', 'points', 'pose', 'flow']
        for data_type in data_types:
            if data_type == 'rgb':
                obs['rgb'] = self.sensor.data.output['tactile_rgb'].squeeze(0)
            elif data_type == 'rgb_marker':
                obs['rgb_marker'] = self.sensor.data.output['marker_rgb'].squeeze(0)
            elif data_type == 'depth':
                obs['depth'] = self.sensor.data.output['height_map'].squeeze(0)
            elif data_type == 'marker':
                obs['marker'] = self.sensor.data.output['marker_motion'].squeeze(0)
            elif data_type == 'points':
                obs['points'] = self.get_init_pts()
            elif data_type == 'pose':
                obs['pose'] = self.get_attach_pose().totensor()
            elif data_type == 'contact_force':
                obs['contact_force'] = self._get_contact_force()
            elif data_type == 'marker_force':
                obs['marker_force'] = self.get_marker_force(mode='interp')
            elif data_type == 'marker_force_scatter':
                obs['marker_force_scatter'] = self.get_marker_force(mode='scatter')
            elif data_type == 'marker_force_img':
                obs['marker_force_img'] = self.get_marker_force_image(mode='interp')
            elif data_type == 'force_field':
                obs['force_field'] = self.get_force_field()
            elif data_type == 'force_field_img':
                obs['force_field_img'] = self.get_force_field_image()
        return obs

    def _get_contact_force(self):
        """[PATCH-A] per-vertex physical contact force (N_v,3) world frame, sparse (contact verts nonzero),
        vertex order aligned with self.gelpad.data.nodal_pos_w."""
        idx, grad = self.uipc_sim.get_contact_gradient()
        offs = self.uipc_sim._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"]
        start = int(offs[self.gelpad.global_system_id])
        num_v = self.gelpad.data.nodal_pos_w.shape[0]
        dense = torch.zeros((num_v, 3), dtype=torch.float32, device=self.device)
        if idx.shape[0] > 0:
            m = (idx >= start) & (idx < start + num_v)
            if m.any():
                loc = torch.as_tensor(idx[m] - start, device=self.device, dtype=torch.long)
                dense[loc] = torch.as_tensor(-grad[m], dtype=torch.float32, device=self.device)
        return dense

    def _world_to_sensor_rot(self):
        """[PATCH-B] Rotation matrix (local->world, 3x3) of the sensor camera frame.

        Same frame the marker flow is computed in, so the resulting force axes line up with the
        marker-flow axes (xy ~ shear, z ~ gel normal). For a force (free vector, no translation):
        f_sensor = f_world @ R.
        """
        cam = self.sensor.camera
        cam._update_poses(cam._ALL_INDICES)
        R = math_utils.matrix_from_quat(cam._data.quat_w_ros)  # (num_envs, 3, 3)
        return R[0]

    def _precompute_marker_force_maps(self):
        """[PATCH-B] Build the marker<->mesh maps used to project per-vertex contact force onto
        the marker grid. Cached on first use (must run after setup() initialised the bindings)."""
        mm = self.sensor.marker_motion_simulator.marker_motion_sim  # VisionTactileSensorUIPC
        if not hasattr(mm, "marker_surf_idx"):
            raise RuntimeError(
                "Marker<->mesh binding not found. get_marker_force only works with the FEM "
                "(ManiSkill) marker simulator, and after TactileManager.setup() has run."
            )
        # barycentric binding: surface-local vertex indices + weights, in the marker-grid order
        self._mf_surf_idx = torch.as_tensor(mm.marker_surf_idx, device=self.device, dtype=torch.long)     # (M,3)
        self._mf_weight = torch.as_tensor(mm.marker_weight, device=self.device, dtype=torch.float32)      # (M,3)
        self._mf_surf_global = torch.as_tensor(mm.vertices_on_surface, device=self.device, dtype=torch.long)  # (N_surf,)
        self._mf_num_markers = int(self._mf_surf_idx.shape[0])

        # Marker positions in the gel/camera xy plane, reconstructed from the same binding so they
        # share the surface vertices' frame. Used to assign a nearest marker to each surface vertex.
        surf_xy = mm.init_surface_vertices_camera[:, :2].to(self.device, dtype=torch.float32)  # (N_surf,2)
        marker_xy = (surf_xy[self._mf_surf_idx] * self._mf_weight[..., None]).sum(1)           # (M,2)
        self._mf_nearest = torch.argmin(torch.cdist(surf_xy, marker_xy), dim=1)                # (N_surf,)

    def get_marker_force(self, mode: str = 'interp', in_sensor_frame: bool = True, reshape: bool = False):
        """[PATCH-B] Map the per-vertex UIPC contact force onto the marker grid.

        Two strategies (use the same marker<->mesh binding as the marker flow):

        - mode='interp' (gather): each marker samples the force field at its location,
          f_m = w0*f0 + w1*f1 + w2*f2 over the 3 vertices it is bound to. Spatially aligned with
          the marker flow, but NOT force-conserving (Sum_markers != Sum_surface) and only "sees"
          the 3 vertices per marker.
        - mode='scatter' (nearest marker): every sensing-surface vertex dumps its full force onto
          its nearest marker. Conserves total force over the sensing surface
          (Sum_markers == Sum_surface), at coarser spatial localization.

        Args:
            mode: 'interp' or 'scatter'.
            in_sensor_frame: if True, rotate force from world frame into the sensor camera frame
                (xy ~ shear, z ~ gel normal). If False, return world-frame force.
            reshape: if True and the marker count matches marker_shape, return
                (marker_shape[1], marker_shape[0], 3); otherwise return (M, 3).

        Returns:
            Tensor of shape (M, 3) (or grid-shaped if reshape=True).
        """
        if not hasattr(self, "_mf_surf_idx"):
            self._precompute_marker_force_maps()

        force_w = self._get_contact_force()          # (N_v, 3) world frame, sparse
        force_surf = force_w[self._mf_surf_global]    # (N_surf, 3)

        if mode == 'interp':
            marker_force = (force_surf[self._mf_surf_idx] * self._mf_weight[..., None]).sum(1)  # (M,3)
        elif mode == 'scatter':
            marker_force = torch.zeros((self._mf_num_markers, 3), dtype=force_surf.dtype, device=self.device)
            marker_force.index_add_(0, self._mf_nearest, force_surf)
        else:
            raise ValueError(f"Unknown marker force mode: {mode!r} (use 'interp' or 'scatter')")

        if in_sensor_frame:
            R = self._world_to_sensor_rot().to(marker_force.dtype)  # local->world
            marker_force = marker_force @ R                          # world -> sensor-local

        if reshape:
            sx, sy = self.sensor.marker_motion_simulator.marker_motion_sim.marker_shape
            if marker_force.shape[0] == sx * sy:
                marker_force = marker_force.reshape(sy, sx, 3)
        return marker_force

    def get_marker_force_image(
        self,
        mode: str = 'interp',
        style: str = 'tacff',
        base: str = None,
        shear_scale: float = None,
        normal_scale: float = None,
    ):
        """[PATCH-B] Render the marker force field as a TacFF-style image (for inspection / video).

        Draws a quiver plot on the regular marker lattice -- one arrow per marker for the in-plane
        shear (Fx, Fy), coloured by the normal force |Fz|. This mirrors the "tactile force field"
        visualization in ContactWorld (arXiv:2606.13877): a grid of arrows on a black background,
        green -> red as contact/normal force grows.

        Args:
            mode: force mapping mode, 'interp' or 'scatter' (see get_marker_force).
            style: 'tacff'   -> green->red arrows on a black background (paper style), or
                   'overlay' -> white arrows + JET dots on the gel image.
            base: background override -- 'black', 'white', 'rgb' (optical) or 'rgb_marker'.
                  Defaults to 'black' for tacff and 'rgb' for overlay.
            shear_scale: pixels per force-unit for the arrows. None -> auto (max arrow ~20px).
            normal_scale: |Fz| mapped to the colour extreme. None -> auto (per-frame max|Fz|).
                Pass a fixed value for frame-to-frame comparable colours.

        Returns:
            (H, W, 3) uint8 torch tensor (RGB).
        """
        import cv2
        mm = self.sensor.marker_motion_simulator.marker_motion_sim
        if not hasattr(self, "_mf_surf_idx"):
            self._precompute_marker_force_maps()

        H, W = mm.tactile_img_height, mm.tactile_img_width
        if base is None:
            base = 'black' if style == 'tacff' else 'rgb'

        # --- background image (H, W, 3) uint8 RGB ---
        if base == 'black':
            img = np.zeros((H, W, 3), dtype=np.uint8)
        elif base == 'white':
            img = np.full((H, W, 3), 255, dtype=np.uint8)
        elif base == 'rgb':
            img = self.sensor.data.output['tactile_rgb'].squeeze(0).cpu().numpy()
        elif base == 'rgb_marker':
            img = self.sensor.data.output['marker_rgb'].squeeze(0).cpu().numpy()
        else:
            raise ValueError(f"Unknown base: {base!r} (use 'black', 'white', 'rgb' or 'rgb_marker')")
        img = np.ascontiguousarray(img.astype(np.uint8))

        # --- force (sensor frame: xy = shear, z = normal), aligned with the marker order ---
        force = self.get_marker_force(mode=mode, in_sensor_frame=True).cpu().numpy()  # (M,3)
        shear = force[:, :2]
        smag = np.linalg.norm(shear, axis=1)
        fz = force[:, 2]

        # --- marker uv = project the reference (undeformed) lattice through the camera ---
        ref_pts = (
            mm.reference_surface_vertices_camera[self._mf_surf_idx].cpu().numpy()
            * self._mf_weight.cpu().numpy()[..., None]
        ).sum(1).astype(np.float32)                                   # (M,3) camera frame
        uv = mm.gen_marker_uv(ref_pts)                                # (M,2) pixels

        # --- per-marker colour from the normal force magnitude ---
        s_n = (np.abs(fz).max() if normal_scale is None else normal_scale)
        s_n = s_n if s_n > 1e-9 else 1.0
        t = np.clip(np.abs(fz) / s_n, 0.0, 1.0)                       # 0 = no normal, 1 = strong
        if style == 'tacff':
            # green (low) -> red (high), RGB
            colors = np.stack([t * 255, (1 - t) * 255, np.zeros_like(t)], axis=1)
        else:  # overlay: JET colormap dots
            cidx = (np.clip(fz / s_n, -1, 1) * 0.5 + 0.5) * 255
            colors = cv2.applyColorMap(cidx.reshape(-1, 1).astype(np.uint8), cv2.COLORMAP_JET)
            colors = colors.reshape(-1, 3)[:, ::-1]                   # BGR -> RGB
        colors = colors.astype(np.uint8)

        # --- shear auto-scale so the largest arrow is ~20px ---
        if shear_scale is None:
            shear_scale = (20.0 / smag.max()) if smag.max() > 1e-9 else 0.0

        for i in range(uv.shape[0]):
            u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
            if not (0 <= u < W and 0 <= v < H):
                continue
            col = tuple(int(c) for c in colors[i])
            du = int(round(shear[i, 0] * shear_scale))
            dv = int(round(shear[i, 1] * shear_scale))
            if style == 'tacff':
                # small base dot + arrow, both in the normal-coloured tone
                cv2.circle(img, (u, v), 1, col, thickness=-1, lineType=cv2.LINE_AA)
                if smag[i] > 1e-9:
                    cv2.arrowedLine(img, (u, v), (u + du, v + dv),
                                    col, 1, line_type=cv2.LINE_AA, tipLength=0.35)
            else:
                cv2.circle(img, (u, v), 3, col, thickness=-1, lineType=cv2.LINE_AA)
                if smag[i] > 1e-9:
                    cv2.arrowedLine(img, (u, v), (u + du, v + dv),
                                    (255, 255, 255), 1, line_type=cv2.LINE_AA, tipLength=0.3)

        return torch.as_tensor(img, dtype=torch.uint8, device=self.device)

    def _precompute_force_field_map(self, grid):
        """[PATCH-C] Build a dense (W x H) sampling grid over the gel surface and bind each grid
        point to the surface mesh by barycentric interpolation (via Delaunay on the reference
        surface). Cached per grid size. grid = (W, H).

        NOTE: this densely RESAMPLES the same per-vertex force field; the real spatial resolution
        is still capped by the gel mesh (a denser grid is interpolation, not new information).
        """
        from scipy.spatial import Delaunay
        mm = self.sensor.marker_motion_simulator.marker_motion_sim
        if not hasattr(self, "_mf_surf_global"):
            self._precompute_marker_force_maps()
        surf_xy = mm.init_surface_vertices_camera[:, :2].cpu().numpy().astype(np.float64)  # (N_surf,2)
        W, H = int(grid[0]), int(grid[1])
        (xmin, ymin), (xmax, ymax) = surf_xy.min(0), surf_xy.max(0)
        GX, GY = np.meshgrid(np.linspace(xmin, xmax, W), np.linspace(ymin, ymax, H))  # (H,W)
        pts = np.stack([GX.ravel(), GY.ravel()], axis=1)                              # (H*W,2)

        tri = Delaunay(surf_xy)
        s = tri.find_simplex(pts)                                # (H*W,), -1 outside the surface
        T = tri.transform[s]
        bc = np.einsum("nij,nj->ni", T[:, :2, :], pts - T[:, 2, :])
        bary = np.concatenate([bc, 1.0 - bc.sum(1, keepdims=True)], axis=1)           # (N,3)
        verts = tri.simplices[s]                                 # (N,3) surface-local vertex idx
        valid = s >= 0
        verts[~valid] = 0
        bary[~valid] = 0.0

        self._ff_grid = (W, H)
        self._ff_verts = torch.as_tensor(verts, device=self.device, dtype=torch.long)    # (N,3)
        self._ff_bary = torch.as_tensor(bary, device=self.device, dtype=torch.float32)   # (N,3)
        self._ff_valid = torch.as_tensor(valid, device=self.device).float()[:, None]     # (N,1)

    def get_force_field(self, grid=(64, 48), in_sensor_frame=True):
        """[PATCH-C] Dense tactile force field: barycentric-interpolate the per-vertex UIPC contact
        force onto a regular grid over the gel surface. Returns (H, W, 3) (rows=H, cols=W), in the
        sensor frame (xy = shear, z = normal) by default.

        This is the same physical force as `marker_force`, just densely resampled to grid=(W,H).
        It does NOT add spatial resolution beyond the gel mesh -- a 64x48 grid interpolates the
        ~60-70 contact vertices up to 3072 cells. For true higher resolution, refine the gel mesh.
        """
        if getattr(self, "_ff_grid", None) != (int(grid[0]), int(grid[1])):
            self._precompute_force_field_map(grid)
        force_surf = self._get_contact_force()[self._mf_surf_global]            # (N_surf,3) world
        fld = (force_surf[self._ff_verts] * self._ff_bary[..., None]).sum(1) * self._ff_valid  # (N,3)
        if in_sensor_frame:
            fld = fld @ self._world_to_sensor_rot().to(fld.dtype)              # world -> sensor
        W, H = self._ff_grid
        return fld.reshape(H, W, 3)

    def get_force_field_image(self, grid=(64, 48), upscale=8, arrow_every=6,
                              normal_scale=None, shear_scale=None):
        """[PATCH-C] Render the dense force field as a TacFF-style image: normal force |fz| as a
        black->green->red intensity map (black = no contact), with sub-sampled white shear arrows.
        Returns an (H*upscale, W*upscale, 3) uint8 RGB tensor.
        """
        import cv2
        fld = self.get_force_field(grid=grid, in_sensor_frame=True).cpu().numpy()  # (H,W,3)
        H, W = fld.shape[:2]
        fz = fld[..., 2]
        shear = fld[..., :2]
        s_n = (np.abs(fz).max() if normal_scale is None else normal_scale)
        s_n = s_n if s_n > 1e-9 else 1.0
        t = np.clip(np.abs(fz) / s_n, 0.0, 1.0)
        # green->red ramp, brightness = activation so no-contact -> black
        img = np.zeros((H, W, 3), np.float32)
        img[..., 0] = t * t * 255.0          # R grows with contact
        img[..., 1] = (1.0 - t) * t * 255.0  # G peaks at light contact
        img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
        img = cv2.resize(img, (W * upscale, H * upscale), interpolation=cv2.INTER_LINEAR)

        smag = np.linalg.norm(shear, axis=2)
        if shear_scale is None:
            mx = smag.max()
            shear_scale = (arrow_every * upscale * 0.9 / mx) if mx > 1e-9 else 0.0
        for r in range(0, H, arrow_every):
            for c in range(0, W, arrow_every):
                if smag[r, c] <= 1e-9:
                    continue
                u, v = int((c + 0.5) * upscale), int((r + 0.5) * upscale)
                du = int(shear[r, c, 0] * shear_scale)
                dv = int(shear[r, c, 1] * shear_scale)
                cv2.arrowedLine(img, (u, v), (u + du, v + dv),
                                (255, 255, 255), 1, line_type=cv2.LINE_AA, tipLength=0.3)
        return torch.as_tensor(img, dtype=torch.uint8, device=self.device)

    def _reset_idx(self):
        self.init_pose_mat = self.get_attach_pose().to_transformation_matrix()
        # self.gelpad.write_vertex_positions_to_sim(vertex_positions=self.gelpad.init_vertex_pos)
    
    def get_min_depth(self):
        return torch.min(self.sensor.data.output['height_map']).item()

class TactileManager:
    def __init__(self, cfg_list: list[TactileCfg], task:'BaseTask', robot_manager=None):
        self.task = task
        self.scene = task.scene
        self.uipc_sim = task.uipc_sim
        # 双臂时可绑定到指定臂; 默认主臂(向后兼容)
        self.robot = (robot_manager or task._robot_manager).robot
        
        self.tactiles = {
            cfg.name: VisualTactileSensor(
                cfg.name, cfg, self.robot, self.scene, self.uipc_sim
            ) for cfg in cfg_list
        }

    def update(self, dt, force_recompute=False):
        for tact in self.tactiles.values():
            tact.update(dt=dt, force_recompute=force_recompute)
 
    def set_debug_vis(self, debug_vis):
        if not debug_vis: return
        for tact in self.tactiles.values():
            tact.set_debug_vis()

    def get_observations(self, data_types: list[str] = None):
        obs = {}
        for name, tact in self.tactiles.items():
            obs[name] = tact.get_observations(data_types)
        return obs

    def get_min_depth(self):
        self.task._update_render()
        depth = []
        for tact in self.tactiles.values():
            depth.append(tact.get_min_depth())
        return torch.tensor(depth, dtype=torch.float32, device=self.task.device)

    def _reset_idx(self):
        for tact in self.tactiles.values():
            tact._reset_idx()

    def setup(self):
        for tact in self.tactiles.values():
            tact.setup()