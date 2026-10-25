"""Export both recordings as 4D arrays for quantem (repo environment): column-flipped to quantem's
canonical layout, since quantem has no column-flip flag; the unbinned one cropped to 128² around
the disk so its float32 copy fits in memory. → recon/quantem_inputs/ (~3.3 GB)."""
import json
import sys

import numpy as np

from paths import INPUTS, RECORDINGS, REPO, cal

sys.path.insert(0, str(REPO))
from tools.calibrate_parallax import file_batches  # noqa: E402

INPUTS.mkdir(parents=True, exist_ok=True)
for mode, size in (("binned", 96), ("unbinned", 128)):
    c = json.load(open(cal(mode)))
    h, w = c["disk"]["shape"]
    r, col = c["disk"]["center"]
    cc = w - 1 - col                                   # the centre's column after the flip
    r0, c0 = (0, 0) if mode == "binned" else (int(round(r)) - 64, int(round(cc)) - 64)
    out = np.lib.format.open_memmap(INPUTS / f"{mode}_canonical.npy", mode="w+", dtype=np.uint16,
                                    shape=(65536, size, size))
    for ids, frames in file_batches(RECORDINGS[mode])(1):
        out[ids] = frames[:, :, ::-1][:, r0:r0 + size, c0:c0 + size]
    out.flush()
    json.dump({"origin": [r - r0, cc - c0], "crop": [r0, c0]},
              open(INPUTS / f"{mode}_canonical.json", "w"))
    print(f"{mode}: {out.shape} written, disk centre at {r - r0:.2f}, {cc - c0:.2f}")
