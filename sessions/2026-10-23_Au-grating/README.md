# Au diffraction grating, 2026-10-23

Shengbo You's session with the new browser viewer: two recordings of an Au grating, live A1
(two-fold astigmatism) tuning, and an upsampling series.

What the recordings show:
- The calibration's C10, fitted at low frequencies, focuses the carbon support.
- The Au islands' lattice is sharpest 136–148 Å (14–15 nm) closer to focus, about one island
  radius above the support.
- Refined on the lattice, the reconstructions resolve Au 111 and 200 from ×2 up.

The recordings are **not in git** (0.8 GB; ask Georgios Varnavides or Berk Kucukoglu). The
layout is [`sessions/README.md`](../README.md)'s; [`analysis/README.md`](analysis/README.md)
lists this session's own scripts and figures.

## The data

| | |
|---|---|
| microscope | 200 keV, ARINA D030056 behind a CEFID energy filter, binned 2×2 (96 × 96, 200 µm pixels), 10 µs frames |
| recordings (`data/Initial_test_A1/`) | `not_centered_…dstream` (armed 16:44, the disk 11 px off-centre) and `Gd2O3_…dstream` (17:12, centred): 128² scans, 16,384 frames each, ~20k counts per frame |
| screen recordings | `data/Initial_test_A1/`: the first live run (17:08), A1 tuning (17:20, 17:38), rotation 3.75° against 4.78° (17:41). `data/upsample_test/`: a 256² scan over 50.7 nm at ×1/×2/×4/×8 (17:58–18:03) |
| viewer exports | `data/Initial_test_A1/snapshot*`: PNGs of the views and two `.npy` (×4 phase-flipped), taken during the A1 tuning |
| live settings | `--accumulate latest`, C10 −430 Å, rotation 4.78° (the meta line); the rest presumed from 2026-10-20 (`session.json`) |

The file names are the operator's template. They say "Gd2O3", but the specimen is Au: the
lattice indexes as Au 111/200. Their "30mrad" and "17.9nm" are nominal.

## Calibration

`calibration/<id>.flags` hold the `run.py` flags. The calibration was made with 1.066 mrad/px,
31.9 mrad and a 1.341 Å step (why below).

| | centered | not centered |
|---|---|---|
| disk centre (row, col), raw frame | (47.18, 46.29) | (37.08, 48.08) |
| disk radius (50% edge) | 29.9 px | 30.0 px |
| columns | flipped | flipped |
| rotation | 4.72° | 5.04° |
| C10, shift fit (the support) | −521 Å | −461 Å |
| **C10, refined on the lattice** (`--defocus`) | **−385 Å** (32× the lattice prominence) | **−313 Å** (24×) |
| C12 | 24 Å at 48° | 17 Å at 27° |

The calibration's area-based disk radius (`summary.json`, 30.9 / 30.7 px) also counts the dark
field that the Au scatters just outside the disk. The 50% edge is the disk itself.

- **The disk is 6.3% larger than on 2026-10-20** (`analysis/disk_size.py`,
  `figures/disk_vs_oct20.jpg`).
  - It measures 29.9 px against 28.1 px, with the same detector, binning and energy. Every 8×8
    tile of the scan gives 29.8–30.0 px, so the disk doesn't wander.
  - So α × the effective camera length at the ARINA changed. The convergence angle (an
    aperture) and the camera length (a preset) should both be fixed; a change in the energy
    filter's projection, or in the objective excitation at another specimen height, would
    explain it.
  - Which of α and the angular sampling changed is not settled. It only affects the C10 and
    C12 reported in Å, not the images: the reconstruction depends only on what is measured in
    pixels.
  - The flags keep Oct 20's 1.066 mrad/px, and 31.9 mrad for the measured disk.
- **The scan step is 1.34 Å, not the nominal 17.9 nm / 128 = 1.398 Å** (`analysis/scan_step.py`,
  `figures/scan_step.jpg`).
  - The live snapshots show Au 111 and 200 at 7.04 and 6.07 output pixels. Their ratio is
    1.160, against 1.155 for Au.
  - This assumes the snapshots were at the recordings' field of view.
- **Cross-check** (`analysis/ctf_zeros.py`, `figures/ctf_zeros.jpg`).
  - The CTF zeros of the support-focus reconstruction fall where the calibration puts them,
    to 2–3% (r = 1.03 and 1.02).
  - On 2026-10-20, r is 0.95–0.96.
  - The fitted C10 itself carries about 3%, so this test can't split angular sampling from the
    step any finer.
- **Beyond quadratic** (a diagnostic only): about +4.8 µm of apparent C30, of the opposite sign
  to 2026-10-20's −5.3 µm.

## Reconstructions

`figures/overview.jpg` shows each recording at the presumed live settings, the support focus and
the lattice focus, its spectrum, and the calibration along C10. `figures/upsampling_<id>.jpg`
shows ×1/×2/×4/×8.

- **The support against the islands.**
  - At the shift fit's C10 the carbon is sharp and the islands show no lattice.
  - At the refined C10 the Au fringes appear, and the spectra show 111 and 200 spots.
  - The variance loss (low frequencies) and the lattice prominence peak 136–148 Å apart. That's
    5–6 depths of field (λ/α² = 25 Å): parallax depth sectioning.
- **The live flags happened to sit near the islands.** −430 Å at 1.066 mrad/px and 1.398 Å is
  about −412 Å in the calibrated units, so the live view showed the lattice.
- **Upsampling.**
  - At ×1 the Au reflections lie beyond the scan's Nyquist frequency (0.37 1/Å) and alias.
  - From ×2 up, 111 and 200 are resolved; crossed fringes appear where grains overlap.
  - ×4 and ×8 look alike at this dose.
- **Astigmatism.** Both recordings predate the A1 tuning, so C12 is 17–24 Å. The flags correct
  it.

## The live session

From the screen recordings and the viewer's meta line:

- **A1 tuning.** Turning the stigmators while watching, the FFT's Thon rings collapse into an X:
  hyperbolic zeros, from strong two-fold astigmatism.
- **Throughput.** 128² passes ran at 40–57 kHz effective, about 0.3 s per pass. The viewer kept
  about 4.6 updates per second up to 2048² (×8).
- **Issues seen.**
  - The viewer server printed a `ConnectionResetError` traceback whenever the browser dropped a
    connection.
  - The Holoscan operators log about 20 `[info]` lines per 2048-frame batch.
  - One final snapshot read "pass 389 at 1%" with exactly 388 passes of frames.
  - The calibration's area-based disk radius is biased by the halo (above).

## Reproducing

With the recordings in `data/` and the repository's environment
(`uv pip install -e ".[local,dev,viz]"`), after the 2026-10-20 session's step 1 (`disk_size.py`
and `ctf_zeros.py` compare with it):

```bash
sessions/2026-10-23_Au-grating/analysis/reproduce.sh      # ~2 min on an M4 Pro
```

It runs:
1. the standard analysis (`scripts/session_report.py`): calibration, reconstructions, overview,
   upsampling figures, `summary.json`;
2. this session's checks: disk size, scan step, CTF zeros.
