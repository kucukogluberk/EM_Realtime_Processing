"""The live OBF (realtime_stem/processing/obf) against quantem, in quantem's environment, on identical
inputs against its correlation route and against its own OBF of the data (quantem_recons.py)
→ recon/obf/<mode>_obf_U<u>.npy and a printed comparison.

    $QUANTEM_PY obf_vs_quantem.py binned [U] [device]    # U = 2, device = mps by default

Identical inputs: the bright-field pixels of run.py's geometry (the calibration with C10 refined, on
the column-flipped frames of export_for_quantem.py), each divided by its scan mean, go to
DirectPtychographyMontage.from_virtual_bfs, and the port's probe sits on the rounded disk centre,
so the two differ only in the port. Both hold only the output grid's band in a patch (the port's
`margin`, quantem's `patch_margin`, 4 pixels per field of view each). Against quantem's own OBF
(recon/quantem/) the port runs as it does live, with the probe on the measured sub-pixel centre
(quantem resamples the frames onto it instead). The bright-field masks still differ at the edge,
and the FFT route wraps the canvas where the correlation route does not, so there only the
interior (an eighth of the field in from each edge) counts. The live image is saved."""
import json
import math
import sys
import time

import numpy as np
import torch

from paths import INPUTS, QUANTEM, RECON, REPO, cal, geometry

sys.path.insert(0, str(REPO))
from realtime_stem.processing.obf.core import ObfSetup, bf_offsets, reconstruct_once  # noqa: E402
from realtime_stem.processing.parallax.engine import preprocess_geometry  # noqa: E402

from quantem.core.datastructures import Dataset2d, Dataset3d  # noqa: E402
from quantem.diffractive_imaging import DirectPtychographyMontage  # noqa: E402

mode = sys.argv[1]
U = int(sys.argv[2]) if len(sys.argv) > 2 else 2
device = sys.argv[3] if len(sys.argv) > 3 else "mps"
c = json.load(open(cal(mode)))
g = geometry(mode)
size = 96 if mode == "binned" else 128
frames = np.load(INPUTS / f"{mode}_canonical.npy", mmap_mode="r")
origin = tuple(json.load(open(INPUTS / f"{mode}_canonical.json"))["origin"])
scan = tuple(c["scan"])
coefs = {"C10": g["c10"], "C12": g["c12"], "phi12": math.radians(g["phi12_deg"])}
geo = preprocess_geometry((size, size), scan, (c["scan_sampling"],) * 2, c["energy"],
                          c["semiangle"], (c["angular_sampling"],) * 2, aberration_coefs=coefs,
                          rotation_angle=math.radians(g["rotation_deg"]), upsampling_factor=U,
                          probe_center=origin)


def sync():
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def corr(a, b, trim=0):
    if trim:
        a, b = a[trim:-trim, trim:-trim], b[trim:-trim, trim:-trim]
    return np.corrcoef(np.ravel(a), np.ravel(b))[0, 1]


t = time.perf_counter()
store = np.empty((frames.shape[0], geo.n_bf), dtype=np.uint16)
for lo in range(0, frames.shape[0], 4096):
    store[lo:lo + 4096] = frames[lo:lo + 4096].reshape(-1, size * size)[:, geo.bf_flat_inds]
print(f"[{mode}] {store.shape[0]} frames, {geo.n_bf} bright-field pixels, gathered in "
      f"{time.perf_counter() - t:.0f} s", flush=True)

# the port: setup (the Gram), then the whole store; on the rounded centre, and as live
held = torch.ones(store.shape[0], device=device)
dev_store = torch.from_numpy(store.view(np.int16)).to(device)
images = {}
for subpixel in (False, True):
    sync(); t = time.perf_counter()
    setup = ObfSetup.build(geo, device=device, subpixel_center=subpixel)
    sync(); t_setup = time.perf_counter() - t
    t = time.perf_counter()
    images[subpixel] = reconstruct_once(dev_store, held, "u16", geo,
                                        setup=setup).cpu().numpy().astype(np.float64)
    sync(); t_mine = time.perf_counter() - t
    print(f"[{mode}] port{' (live)' if subpixel else ''}: {setup.placement}, canvas "
          f"{setup.inner_shape} ({setup.multiples} fields of view of {setup.per_fov} px), patch "
          f"{setup.grid_shape}, setup {t_setup:.1f} s, {t_mine:.1f} s = "
          f"{t_mine / store.shape[0] * 1e6:.0f} us/frame", flush=True)
del dev_store, setup
mine, live = images[False], images[True]
(RECON / "obf").mkdir(exist_ok=True)
np.save(RECON / "obf" / f"{mode}_obf_U{U}.npy", live.astype(np.float32))

# quantem's correlation route on the same bright-field pixels and values
off = bf_offsets(geo)
mask = np.zeros((size, size), bool)
mask[off[:, 0] % size, off[:, 1] % size] = True
order = np.lexsort((off[:, 1] % size, off[:, 0] % size))      # the mask's row-major order
vbf = store[:, order].astype(np.float32)
vbf /= vbf.mean(0)
vbf = np.ascontiguousarray(vbf.T).reshape(-1, *scan)
del store
dk = c["angular_sampling"] * 1e-3 / geo.wavelength
model = DirectPtychographyMontage.from_virtual_bfs(
    Dataset3d.from_array(vbf, sampling=(1, c["scan_sampling"], c["scan_sampling"]),
                         units=("index", "A", "A")),
    Dataset2d.from_array(mask, sampling=(dk, dk), units=("A^-1", "A^-1")),
    energy=c["energy"], rotation_angle=g["rotation_deg"], semiangle_cutoff=c["semiangle"],
    aberration_coefs=coefs, device=device, verbose=False)
del vbf
sync(); t = time.perf_counter()
model.precompute_gram(upsampling_factor=U)
sync(); t_qgram = time.perf_counter() - t
t = time.perf_counter()
model.reconstruct(deconvolution_kernel="obf", convolution_mode="correlation", upsampling_factor=U,
                  verbose=False)
theirs = np.asarray(model.obj, dtype=np.float64)
sync(); t_q = time.perf_counter() - t
print(f"[{mode}] quantem's correlation route: Gram {t_qgram:.1f} s, {t_q:.1f} s", flush=True)

trim = mine.shape[0] // 8
err = np.linalg.norm(mine - theirs) / np.linalg.norm(theirs)
print(f"[{mode}] U={U}, identical inputs: relative difference {err:.2e}, correlation "
      f"{corr(mine, theirs):.6f}, std ratio {mine.std() / theirs.std():.5f}")
for route in ("montage", "direct"):
    f = QUANTEM / f"{mode}_{route}_obf_U{U}.npy"
    if f.exists():
        q = np.load(f)
        print(f"[{mode}] U={U}, against quantem's {route} OBF of the data: correlation "
              f"{corr(live, q, trim):.4f} in the interior ({corr(mine, q, trim):.4f} on the "
              f"rounded centre), {corr(live, q):.4f} overall")
