"""Add the markerless "gel + speckle" tactile image (`gel_particle`) to an ALREADY-COLLECTED
hdf5 — no re-simulation. The particle-layer MOTION is driven by the stored marker flow (the
markers are real mesh-bound material points), the silver base comes from the clean optical `rgb`,
and the contact shading from `depth`. So it is a faithful offline reconstruction of the same
representation `get_gel_particle_image` produces in-sim (see envs/sensors/tactile.py, PATCH-G).

Requires per gel: `rgb` (clean Taxim gel), `marker` (T,2,64,2 marker motion). `depth` optional.
Writes `tactile/<gel>/gel_particle` as (T,240,320,3) uint8, matching the in-sim format/resolution.

Usage:
  # preview a 6-phase contact sheet (no write) to eyeball the look:
  python add_gel_particle_offline.py IN.hdf5 --preview /tmp/gp_offline.png
  # add gel_particle into a COPY (non-destructive), or --inplace to edit IN.hdf5:
  python add_gel_particle_offline.py IN.hdf5 --out OUT.hdf5
  python add_gel_particle_offline.py IN.hdf5 --inplace
"""
import argparse, shutil, numpy as np, cv2, h5py

ap = argparse.ArgumentParser()
ap.add_argument("hdf5")
ap.add_argument("--gels", default="left_tactile,right_tactile")
ap.add_argument("--preview", default=None, help="write a 6-phase sheet PNG and exit (no hdf5 write)")
ap.add_argument("--out", default=None, help="write result to this new hdf5 (copy of input + gel_particle)")
ap.add_argument("--inplace", action="store_true", help="add gel_particle into the input hdf5 in place")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def dec(b):
    a = cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def build_coating(H, W, seed):
    rs = np.random.RandomState(seed)
    n = int(0.09 * H * W)
    xs = rs.uniform(0, W, n); ys = rs.uniform(0, H, n)
    hue = rs.uniform(0, 180, n).astype(np.uint8); sat = rs.uniform(120, 210, n).astype(np.uint8)
    val = rs.uniform(200, 255, n).astype(np.uint8)
    cols = cv2.cvtColor(np.stack([hue, sat, val], 1)[None], cv2.COLOR_HSV2RGB)[0].astype(np.float32)
    coat = np.zeros((H, W, 3), np.float32); alpha = np.zeros((H, W), np.float32)
    for x, y, c, a in zip(xs, ys, cols, rs.uniform(0.6, 1.0, n)):
        cv2.circle(coat, (int(x), int(y)), 1, c.tolist(), -1, cv2.LINE_AA)
        cv2.circle(alpha, (int(x), int(y)), 1, float(a), -1, cv2.LINE_AA)
    coat = cv2.GaussianBlur(coat, (0, 0), 0.5); alpha = cv2.GaussianBlur(alpha, (0, 0), 0.5)
    grain = cv2.GaussianBlur(np.random.RandomState(seed + 1).randn(H, W).astype(np.float32), (0, 0), 0.7)
    grain /= grain.std() + 1e-6
    return coat, alpha, grain


class GelRenderer:
    """Reconstruct gel_particle for one gel: silver rgb base + marker-flow advected coating."""
    def __init__(self, rgb0, marker, H, W, seed):
        self.H, self.W = H, W
        self.coat, self.alpha, self.grain = build_coating(H, W, seed)
        self.MX, self.MY = np.meshgrid(np.arange(W), np.arange(H))
        # pick the moving slot (max temporal variance) as 'current'; frame-0 of it = reference
        var = np.array([marker[:, s].std(0).mean() for s in range(marker.shape[1])])
        self.cur = int(var.argmax())
        self.ref_uv = marker[0, self.cur].astype(np.float32)          # (64,2) reference positions
        self.marker = marker

    def disp_field(self, t):
        d = self.marker[t, self.cur].astype(np.float32) - self.ref_uv  # (64,2) per-marker flow
        d = d - d.mean(0, keepdims=True)                               # drop rigid motion
        s = 8; dh, dw = self.H // s, self.W // s
        ax = np.zeros((dh, dw), np.float32); ay = np.zeros((dh, dw), np.float32); cn = np.zeros((dh, dw), np.float32)
        xi = np.clip((self.ref_uv[:, 0] / s).astype(int), 0, dw - 1)
        yi = np.clip((self.ref_uv[:, 1] / s).astype(int), 0, dh - 1)
        np.add.at(ax, (yi, xi), d[:, 0]); np.add.at(ay, (yi, xi), d[:, 1]); np.add.at(cn, (yi, xi), 1.0)
        wc = cv2.GaussianBlur(cn, (0, 0), 2.0) + 1e-3
        DX = cv2.resize(cv2.GaussianBlur(ax, (0, 0), 2.0) / wc, (self.W, self.H))
        DY = cv2.resize(cv2.GaussianBlur(ay, (0, 0), 2.0) / wc, (self.W, self.H))
        return DX.astype(np.float32), DY.astype(np.float32)

    def frame(self, rgb, depth, t):
        H, W = self.H, self.W
        lum = cv2.GaussianBlur(rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32), (0, 0), 1.5)
        g0 = np.clip(172.0 + (lum - lum.mean()) * 1.1, 128, 214)
        gel = np.stack([g0 * 0.985, g0 * 1.0, g0 * 1.03], -1) + self.grain[..., None] * 3.0
        DX, DY = self.disp_field(t)
        mapx = (self.MX - DX).astype(np.float32); mapy = (self.MY - DY).astype(np.float32)
        sp = cv2.remap(self.coat, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        al = cv2.remap(self.alpha, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        a = np.clip(al * 0.45, 0, 1)[..., None]
        out = gel * (1.0 - a) + sp * a
        if depth is not None:                                          # contact shading from indentation
            dd = depth.astype(np.float32); dd = np.clip((dd - dd.min()) / (np.ptp(dd) + 1e-6), 0, 1)
            out = out * (1.0 - 0.10 * dd[..., None])
        return np.clip(out, 0, 255).astype(np.uint8)


def render_gel(f, gel, seed, preview_only):
    g = f["tactile/" + gel]
    rgb_raw = g["rgb"][:]; marker = g["marker"][:]
    depth = g["depth"] if "depth" in g else None
    T = len(rgb_raw)
    H, W = dec(rgb_raw[0]).shape[:2]
    R = GelRenderer(dec(rgb_raw[0]), marker, H, W, seed)
    if preview_only:
        # 6 phases by marker-flow magnitude (proxy for contact)
        mag = np.array([np.abs(marker[t, R.cur] - R.ref_uv).mean() for t in range(T)])
        pk = int(mag.argmax())
        idx = [0, pk // 2, pk, min(T - 1, pk + (T - pk) // 3), min(T - 1, pk + 2 * (T - pk) // 3), T - 1]
        labs = ["approach", "contact", "PEAK", "hold", "ease", "release"]
        tiles = []
        for i, lab in zip(idx, labs):
            im = R.frame(dec(rgb_raw[i]), depth[i] if depth is not None else None, i)
            cv2.putText(im, f"{lab} f{i}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(im, f"{lab} f{i}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
            tiles.append(im)
        sheet = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
        return sheet, None
    out = np.empty((T, H, W, 3), np.uint8)
    for t in range(T):
        out[t] = R.frame(dec(rgb_raw[t]), depth[t] if depth is not None else None, t)
    return None, out


gels = [x for x in args.gels.split(",") if x]
if args.preview:
    with h5py.File(args.hdf5, "r") as f:
        gel = gels[0] if ("tactile/" + gels[0]) in f else list(f["tactile"].keys())[0]
        sheet, _ = render_gel(f, gel, args.seed, True)
    cv2.imwrite(args.preview, sheet[:, :, ::-1])
    print("preview saved:", args.preview, "gel:", gel)
else:
    dst = args.hdf5 if args.inplace else args.out
    assert dst, "give --out OUT.hdf5 or --inplace"
    if not args.inplace:
        shutil.copyfile(args.hdf5, dst)
    with h5py.File(dst, "a") as f:
        for gel in gels:
            node = "tactile/" + gel
            if node not in f:
                continue
            if "gel_particle" in f[node]:
                del f[node]["gel_particle"]
            _, arr = render_gel(f, gel, args.seed, False)
            f[node].create_dataset("gel_particle", data=arr, compression="gzip", compression_opts=4)
            print(f"  wrote {node}/gel_particle {arr.shape}")
    print("done:", dst)
