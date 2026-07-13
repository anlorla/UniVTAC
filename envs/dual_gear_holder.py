from ._base_task import *
import numpy as np

# ============================================================================
# Dual Gear Holder
#   双臂任务: 齿轮散放在 holder 周围, 目标是把齿轮放到 holder 的立柱上。
#   gear       : GEAR.usd        (齿轮平放, 中心孔沿 local Z)
#   holder     : GEAR_HOLDER.usd (圆盘 + 三根竖直立柱)
# ============================================================================

UPRIGHT = [1, 0, 0, 0]

TABLE_TOP = 0.004
GEAR_THICKNESS = 0.016
HOLDER_HEIGHT = 0.0355
HOLDER_PLATE_H = 0.008
PEG_LAYOUT_R = 0.032
GEAR_Z = TABLE_TOP + GEAR_THICKNESS * 0.5
HOLDER_Z = TABLE_TOP + HOLDER_HEIGHT * 0.5
SETTLED_Z = TABLE_TOP + HOLDER_PLATE_H + GEAR_THICKNESS * 0.5 + 0.001
# Release height: keep the gear's bore THREADED on the peg before opening the
# gripper. The old value (= holder top + half-gear + 1.5mm) left the gear bottom
# ~1.5mm ABOVE the peg tip, so the gripper opened with the gear floating above
# the peg -> it free-dropped onto the tip and rolled off (~40mm radial error).
# Lower it to just above the settled height so the peg is fully engaged in the
# bore at release; the gear then slides the last few mm down the peg to settle.
# (The motion planner ignores holder/gears, so the deep descent threads the peg
# via sim contact instead of failing to plan.)
# Release the gear a bit above the holder seat (not plunged all the way down): the
# peg is still fully through the bore here (gear top below the peg top), so the gear
# self-centers as it slides down -- accuracy is kept while the gripper stays clear.
RELEASE_Z = SETTLED_Z + 0.009
PEG_TOP_Z = TABLE_TOP + HOLDER_HEIGHT
# Gear-center height for hovering over a peg before lowering: the bore must clear
# the peg tip, i.e. gear bottom (= center - half-thickness) above PEG_TOP_Z.
HOVER_Z = PEG_TOP_Z + GEAR_THICKNESS * 0.5 + 0.015

GRASP_Z_BIAS = -0.002  # grasp a bit lower on the gear (was +0.002, now 4mm lower)
PRE_GRASP_UP = 0.08
LIFT_UP = 0.055
GRIP_CLOSE = 0.5
RELEASE_RETREAT_UP = 0.06
RELEASE_RETREAT_SIDE = 0.30
CLEAN_MOVE_CONSTRAINT = [1, 1, 1, 1, 1, 0]
CLEAN_MOVE_SPEED = 0.5
HORIZONTAL_MOVE_SPEED = 0.5
PLACE_APPROACH_UP = 0.02
PLACE_ALIGN_TOL = 0.0005

HOLDER_POSE = Pose([0.50, 0.0, HOLDER_Z], UPRIGHT)
GEAR_POSES = [
    Pose([0.36, 0.20, GEAR_Z], UPRIGHT),
    Pose([0.36, -0.20, GEAR_Z], UPRIGHT),
    Pose([0.36, 0.00, GEAR_Z], UPRIGHT),
]
GEAR_NOISE = 0.008

HOLDER_COLOR = (0.10, 0.10, 0.10)
GEAR_COLORS = [
    (0.62, 0.62, 0.58),
    (0.68, 0.66, 0.60),
    (0.56, 0.58, 0.58),
]
PEG_LOCAL_XY = [
    (PEG_LAYOUT_R * np.cos(np.deg2rad(90.0)), PEG_LAYOUT_R * np.sin(np.deg2rad(90.0))),
    (PEG_LAYOUT_R * np.cos(np.deg2rad(210.0)), PEG_LAYOUT_R * np.sin(np.deg2rad(210.0))),
    (PEG_LAYOUT_R * np.cos(np.deg2rad(330.0)), PEG_LAYOUT_R * np.sin(np.deg2rad(330.0))),
]


@configclass
class TaskCfg(BaseTaskCfg):
    dual_arm = True
    viewer = ViewerCfg(eye=(1.35, 0.0, 0.95), lookat=(0.50, 0.0, 0.10))
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
    # Three grasp-place cycles across two arms run ~1500 steps; the defaults
    # (step_lim 1200 / max_save_frames 1000) cut the episode off mid-gear-1.
    step_lim = 2400
    max_save_frames = 2000
    reset_time_limit = 600.0


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        # Manual fixed grip: disable adaptive (tactile-depth) grasping so close_gripper
        # drives the fingers to a fixed commanded opening (GRIP_CLOSE) and squeezes the
        # gear firmly instead of stopping at first contact.
        cfg.use_adaptive_grasp = False
        self.planner_ignore_actors = {"holder", "gear_0", "gear_1", "gear_2"}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {"holder", "gear_0", "gear_1", "gear_2"}
        self._robot_manager_b.ignore_actors = {"holder", "gear_0", "gear_1", "gear_2"}

    def create_actors(self):
        self.holder = self._actor_manager.add_from_usd_file(
            name="holder",
            asset_path="GEAR_HOLDER.usd",
            pose=HOLDER_POSE,
            density=1e5,
        )
        self.gears = [
            self._actor_manager.add_from_usd_file(
                name=f"gear_{i}",
                asset_path="GEAR.usd",
                pose=pose,
                density=8e3,
            )
            for i, pose in enumerate(GEAR_POSES)
        ]

    def _reset_actors(self):
        self.holder.set_color(HOLDER_COLOR, name="HolderDark")
        self.holder.set_pose(HOLDER_POSE)
        self.holder_pose = HOLDER_POSE

        gear_poses = []
        for i, gear in enumerate(self.gears):
            pose = GEAR_POSES[i].add_offset(self.create_noise([GEAR_NOISE, GEAR_NOISE, 0.0]))
            gear.set_color(GEAR_COLORS[i], name=f"GearColor{i}")
            gear.set_pose(pose)
            gear_poses.append(pose.tolist())

        self.metadata["holder_pose"] = HOLDER_POSE.tolist()
        self.metadata["gear_poses"] = gear_poses
        self.metadata["target_poses"] = [pose.tolist() for pose in self._target_poses()]

    def _target_poses(self):
        hp = getattr(self, "holder_pose", HOLDER_POSE)
        return [
            Pose([hp.p[0] + dx, hp.p[1] + dy, RELEASE_Z], UPRIGHT)
            for dx, dy in PEG_LOCAL_XY
        ]

    def _settled_poses(self):
        hp = getattr(self, "holder_pose", HOLDER_POSE)
        return [
            Pose([hp.p[0] + dx, hp.p[1] + dy, SETTLED_Z], UPRIGHT)
            for dx, dy in PEG_LOCAL_XY
        ]

    def _top_grasp(self, actor, rm, atom, arm, camera_up=(1, 0, 0)):
        if not self.move(atom.open_gripper(1.0), arm=arm, delay=False):
            return False
        actor_name = actor.cfg.name
        ap = actor.get_pose()
        grasp_gc = construct_grasp_pose(
            np.array([ap.p[0], ap.p[1], ap.p[2] + GRASP_Z_BIAS]),
            np.array([0.0, 0.0, 1.0]),
            np.array(camera_up, dtype=float),
        )
        # Single grasp move with a grasp-approach metric (pre_dis): curobo plans to a
        # point PRE_GRASP_UP above the grasp, then descends in a straight line along
        # the approach axis. The old code issued two SEPARATE free plans (to pre_gc,
        # then to grasp_gc); each free plan could pick a different wrist IK branch, so
        # the arm visibly swung/reoriented the wrist before descending onto the gear.
        grasp_ee = rm.gripper_center_to_ee(grasp_gc)
        if not self.move(
            [Action("move", target_pose=grasp_ee, pre_dis=PRE_GRASP_UP)],
            arm=arm,
            tag=f"{actor_name}_grasp",
            delay=False,
        ):
            return False
        if not self.move(atom.close_gripper(GRIP_CLOSE), arm=arm, tag=f"{actor_name}_close", delay=False):
            return False
        return self.move(
            atom.move_by_displacement(z=LIFT_UP, xyz_coord="world"),
            arm=arm,
            tag=f"{actor_name}_lift",
            constraint_pose=CLEAN_MOVE_CONSTRAINT,
            time_dilation_factor=CLEAN_MOVE_SPEED,
            delay=False,
        )

    def _move_gripper_center(self, rm, atom, arm, pose, tag, constraint_pose=None, time_dilation_factor=None):
        return self.move(
            atom.move_to_pose(rm.gripper_center_to_ee(pose)),
            arm=arm,
            tag=tag,
            constraint_pose=constraint_pose,
            time_dilation_factor=time_dilation_factor,
            delay=False,
        )

    def _place_on_peg(self, actor, rm, atom, arm, target_pose, tag):
        # Simple, direct placement: the gear was just grasped at its center (top
        # grasp), so the gripper center sits right over the gear center. Capture that
        # small in-hand offset ONCE now while it is reliable and hold it fixed for the
        # whole placement -- never re-read the gear pose mid-place (that is what made
        # the old code aim at corrupted/unreachable targets).
        #   1) carry the held gear to a hover directly above the peg,
        #   2) lower it straight down so the bore threads onto the stick,
        #   3) open the gripper to release it onto the peg,
        #   4) retreat straight up, clear of the peg.
        gp = actor.get_pose()
        gc = rm.get_gripper_center_pose()
        offset = np.array(gc.p, dtype=float) - np.array(gp.p, dtype=float)
        px, py = float(target_pose.p[0]), float(target_pose.p[1])

        def gc_over_peg(gear_z):
            # gripper-center pose that puts the gear center at (px, py, gear_z)
            return Pose([px + offset[0], py + offset[1], gear_z + offset[2]], gc.q)

        # 1. Hover directly above the peg.
        if not self._move_gripper_center(
            rm, atom, arm, gc_over_peg(HOVER_Z), f"{tag}_hover",
            time_dilation_factor=HORIZONTAL_MOVE_SPEED,
        ):
            return False

        # 2. Lower straight down onto the stick (xy + orientation locked). The planner
        #    ignores holder/gears, so the bore threads the peg via sim contact.
        if not self._move_gripper_center(
            rm, atom, arm, gc_over_peg(RELEASE_Z), f"{tag}_lower",
            constraint_pose=CLEAN_MOVE_CONSTRAINT,
            time_dilation_factor=CLEAN_MOVE_SPEED,
        ):
            return False

        # 3. Release the gear onto the peg.
        if not self.move(atom.open_gripper(1.0), arm=arm, tag=f"{tag}_release", delay=False):
            return False

        # 4. Retreat straight up, clear of the peg.
        if not self.move(
            atom.move_by_displacement(z=RELEASE_RETREAT_UP, xyz_coord="world"),
            arm=arm,
            tag=f"{tag}_retreat",
            constraint_pose=CLEAN_MOVE_CONSTRAINT,
            time_dilation_factor=CLEAN_MOVE_SPEED,
            delay=False,
        ):
            return False

        # 5. Move clear of the middle so this arm is out of the way after dropping the
        #    gear: slide horizontally toward this arm's own side (+y for arm A, -y for
        #    arm B) via a relative displacement (robust, like the retreat). This is
        #    cosmetic -- the gear is already released -- so if curobo can't plan it from
        #    here, restore plan_success and carry on instead of aborting the episode.
        side = 1.0 if arm == "a" else -1.0
        if not self.move(
            atom.move_by_displacement(y=side * RELEASE_RETREAT_SIDE, xyz_coord="world"),
            arm=arm,
            tag=f"{tag}_clear",
            time_dilation_factor=HORIZONTAL_MOVE_SPEED,
            delay=False,
        ):
            self.plan_success = True
        return True

    def _gear_dbg(self, tag):
        poses = [gear.get_pose() for gear in self.gears]
        print(
            "[GEAR] "
            + tag
            + " "
            + " ".join(
                f"g{i}=({p.p[0]:.3f},{p.p[1]:.3f},{p.p[2]:.3f})"
                for i, p in enumerate(poses)
            ),
            flush=True,
        )

    def pre_move(self):
        self.delay(5)

    def _play_once(self):
        targets = self._target_poses()

        if not self._top_grasp(self.gears[0], self._robot_manager, self.atom_a, "a", camera_up=(1, 0, 0)):
            return False
        self._gear_dbg("A grasped gear_0")
        if not self._place_on_peg(self.gears[0], self._robot_manager, self.atom_a, "a", targets[0], "gear_0_place"):
            return False
        self._gear_dbg("A placed gear_0")

        if not self._top_grasp(self.gears[1], self._robot_manager_b, self.atom_b, "b", camera_up=(1, 0, 0)):
            return False
        self._gear_dbg("B grasped gear_1")
        if not self._place_on_peg(self.gears[1], self._robot_manager_b, self.atom_b, "b", targets[1], "gear_1_place"):
            return False
        self._gear_dbg("B placed gear_1")

        if not self._top_grasp(self.gears[2], self._robot_manager, self.atom_a, "a", camera_up=(1, 0, 0)):
            return False
        self._gear_dbg("A grasped gear_2")
        if not self._place_on_peg(self.gears[2], self._robot_manager, self.atom_a, "a", targets[2], "gear_2_place"):
            return False
        self._gear_dbg("A placed gear_2")
        self.delay(5, is_save=True)

    def check_success(self):
        targets = self._settled_poses()
        ok = []
        for i, gear in enumerate(self.gears):
            gp = gear.get_pose()
            tp = targets[i]
            radial = float(np.linalg.norm(np.array(gp.p[:2]) - np.array(tp.p[:2])))
            z_err = float(abs(gp.p[2] - tp.p[2]))
            up = float(np.dot(gp.to_transformation_matrix()[:3, 2], np.array([0.0, 0.0, 1.0])))
            placed = radial < 0.012 and z_err < 0.018 and up > 0.75
            self.metadata[f"gear_{i}_radial"] = radial
            self.metadata[f"gear_{i}_z_err"] = z_err
            self.metadata[f"gear_{i}_up"] = up
            ok.append(placed)
            print(
                f"[GEAR] gear_{i}: radial={radial * 1000:.1f}mm z_err={z_err * 1000:.1f}mm "
                f"up={up:.3f} placed={placed}",
                flush=True,
            )
        pa = self._robot_manager.get_gripper_percentage()
        pb = self._robot_manager_b.get_gripper_percentage()
        released = pa > 0.9   # A (final placer) released; B idle-neutral ~0.5 after releasing gear_1 earlier
        print(f"[GEAR] gripperA={pa:.2f} gripperB={pb:.2f} released={released}", flush=True)
        return bool(all(ok) and released)
