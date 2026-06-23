"""Headless-render a USD asset to a PNG for quick visual inspection.

Unlike view_usd.py (interactive GUI, needs a window/swapchain), this renders
offscreen via Replicator, so it works over a headless/greeter X server.

Usage:
    python scripts/render_usd.py assets/objects/BALL.usd [out.png]
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render a USD asset to a PNG.")
parser.add_argument("usd", nargs="?", default="assets/objects/BALL.usd")
parser.add_argument("out", nargs="?", default="/tmp/render_usd.png")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import imageio.v2 as imageio
import omni.replicator.core as rep
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

usd_path = os.path.abspath(args_cli.usd)
print(f"[render_usd] opening: {usd_path}")

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cpu"))

# ground + light
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)).func(
    "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
)
# a warm distant light to bring out the sphere's curvature
sim_utils.DistantLightCfg(intensity=1500.0, color=(1.0, 0.95, 0.9)).func(
    "/World/Sun", sim_utils.DistantLightCfg(intensity=1500.0, color=(1.0, 0.95, 0.9))
)

# spawn the asset, lifted so it rests above the ground
cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
cfg.func("/World/Asset", cfg, translation=(0.0, 0.0, 0.03))

# offscreen camera framed on a ~cm-scale object
cam = rep.create.camera(position=(0.12, 0.12, 0.09), look_at=(0.0, 0.0, 0.02))
rp = rep.create.render_product(cam, (1280, 960))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)

sim.reset()
# step a few times so the renderer accumulates a clean frame
for _ in range(60):
    sim.step()
    rep.orchestrator.step()

img = rgb.get_data()
img = np.asarray(img)[..., :3]
imageio.imwrite(args_cli.out, img)
print(f"[render_usd] saved {args_cli.out}  shape={img.shape}")

simulation_app.close()
