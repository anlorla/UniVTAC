"""Build a self-contained CUP.usd directly with pxr, mirroring the structure of the
repo's working assets (e.g. SLEEVE.usd): /CUP Xform (translate/orient/scale double) +
/CUP/mesh Mesh (points/faceVertexCounts/faceVertexIndices/extent/subdivisionScheme +
physics approx + tet_* attrs). Avoids the asset-converter (which authored a scalar
xformOp:scale that broke fabric sync) and tetgen (which can't tet the thin shell).

Usage (pxr lives in isaacsim extscache):
  EXT=~/miniconda3/envs/UniVTAC/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-*
  PYTHONPATH=$EXT LD_LIBRARY_PATH=$EXT/bin:$EXT/pxr/.libs \
    python scripts/asset_tools/build_cup_usd.py <in.obj> <tet.npz> <out.usd>
"""
import sys, numpy as np, trimesh
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Vt, Gf

obj_path, npz_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]


def bind_material(stage, root_path, mesh_prim, color, name="Surf"):
    """Bind a UsdPreviewSurface so the asset renders lit/colored. The UIPC loader replaces the
    mesh geometry with the tet surface at load time but does NOT touch the material binding,
    so this binding survives and gives the in-sim asset its color (otherwise: flat dark gray)."""
    UsdGeom.Scope.Define(stage, f"{root_path}/Looks")
    mat = UsdShade.Material.Define(stage, f"{root_path}/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"{root_path}/Looks/{name}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(mat)

m = trimesh.load(obj_path, process=False)
if isinstance(m, trimesh.Scene):
    m = m.to_geometry()
V = np.asarray(m.vertices, np.float32)          # meters
F = np.asarray(m.faces, np.int32).reshape(-1, 3)
d = np.load(npz_path)

stage = Usd.Stage.CreateNew(out_path)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

# root Xform with double-precision translate/orient/scale (matches SLEEVE.usd)
xform = UsdGeom.Xform.Define(stage, "/CUP")
xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 1, 1))
stage.SetDefaultPrim(xform.GetPrim())

# mesh
mesh = UsdGeom.Mesh.Define(stage, "/CUP/mesh")
mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.reshape(-1)))
mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(
    np.stack([V.min(0), V.max(0)]).astype(np.float32)))

# collision (mirror SLEEVE: convexDecomposition + enabled)
prim = mesh.GetPrim()
api = UsdPhysics.MeshCollisionAPI.Apply(prim)
api.CreateApproximationAttr().Set("convexDecomposition")
UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)

# embedded tet (fTetWild)
def v3(n, a): prim.CreateAttribute(n, Sdf.ValueTypeNames.Float3Array).Set(Vt.Vec3fArray.FromNumpy(a.reshape(-1, 3).astype(np.float32)))
def ui(n, a): prim.CreateAttribute(n, Sdf.ValueTypeNames.UIntArray).Set(Vt.UIntArray.FromNumpy(a.reshape(-1).astype(np.uint32)))
v3("tet_points", d["tet_points"]); ui("tet_indices", d["tet_indices"])
v3("tet_surf_points", d["surf_points"]); ui("tet_surf_indices", d["surf_indices"])

# material so the cup renders as a lit cream paper cup (not flat dark gray)
bind_material(stage, "/CUP", prim, (0.93, 0.90, 0.84))

stage.GetRootLayer().Save()
print(f"[build_cup_usd] {out_path}: {len(V)} verts / {len(F)} faces / {len(d['tet_indices'])//4} tets, self-contained")
