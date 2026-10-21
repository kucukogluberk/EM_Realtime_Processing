# Gd₂O₃ nanoparticles, 30 mrad, 2026-10-20

The first live session of realtime-stem on the ARINA, and what we learned from its recordings
afterwards: the measured geometry, offline reconstructions (checked against quantem), the live
view part-way through a scan, the continuous-scanning modes, and a focus-walk demo.

The recordings are **not in git** (0.7 GB; ask Georgios Varnavides or Berk Kucukoglu). This folder
holds the scripts that regenerate everything from them, the calibrations, and a few figures.
[`analysis/README.md`](analysis/README.md) lists every script and figure. For the tools themselves
(calibration, looping a recording, the browser viewer), see [`docs/WORKFLOWS.md`](../docs/WORKFLOWS.md).

## The data

Settings: 200 keV, convergence semiangle 30 mrad, ARINA (D030056), 256 × 256 scan over 50.6 nm
(1.9766 Å step).

| file (not in git) | what |
|---|---|
| `Gd2O3_50.6nmFOV_256x256_30mrad_-30nm_binned_10us.dstream` | 65,536 frames, 96 × 96 (2×2 binned) uint16, 10 µs; ~10k counts/frame; 1.066 mrad/px |
| `Gd2O3_50.6nmFOV_256x256_30mrad_-30nm_unbinned.dstream` | 65,536 frames, 192 × 192 uint16, 34 µs; ~29k counts/frame; 0.533 mrad/px; recorded 11 min before the binned scan |
| `…_HAADF.npy` (×2) | 256² HAADF images of the same scans. At these dwell times they are mostly shot noise. |
| `recon_20261020_*.mkv` (×6) | Screen recordings of the live reconstructions (`--accumulate latest --continuous`). The scan, upsampling and dwell of each are in its name. |

The live settings were:
- `--defocus=-300` (the microscope's "−30 nm");
- `--rotation=3.75 --detector-layout rows,-cols`, with the disk centre left at its default;
- the frame mean kept (`--keep-bf-mean` then, the default now).

## Geometry, measured from the recordings

`tools/calibrate_parallax.py` cross-correlates the virtual bright-field images of small groups of
bright-field pixels and fits the engine's shift model, M = (C10·I + C12·Q(φ12))·R(θ)·diag(1, ±1).
Shifts are measured below 0.18 Å⁻¹, from pixels clear of the disk edge by that frequency: near the
edge only one sideband is transferred, which moves the correlation peak outwards. The figure is
`analysis/figures/calibration_binned.png`.

| | live | binned | unbinned |
|---|---|---|---|
| disk centre (row, col), raw frame | default (48, 47) / (96, 95) | (47.86, 48.41) | (96.24, 97.32) |
| disk radius | | 28.33 px = 30.2 mrad | 56.57 px = 30.2 mrad |
| columns | flipped (`rows,-cols`) | flipped | flipped |
| rotation | 3.75° | 4.78 ± 0.09° | 4.77 ± 0.08° |
| C10 | −300 Å | −315.6 ± 0.9 Å | −296.1 ± 0.8 Å |
| C12 | — | 16.7 Å at −7.1° | 12.9 Å at −8.1° |

`analysis/cal_*.flags` hold these as `run.py` flags.

- **Twin and sign.** The shifts can't tell (C10, θ) from (−C10, θ + 180°). Taking the rotation
  near the live 3.75°, C10 is negative, i.e. underfocus in abTEM's convention, as "−30 nm"
  says.
- **Drift.** The 20 Å between the two scans is presumably focus drift over the 11 minutes between
  them. The specimen also moved (+4.9, −6.5) Å.
- **Defective pixels.** All lie outside the disk: three pixels flagged 65535 unbinned, binned
  into two hot pixels. Nothing needs masking.
- **What the live run had wrong:** the rotation by 1°, and the default centre by 1.5–2.5 px in
  columns. The centre error mostly moves the whole image.
- **Beyond quadratic (a diagnostic only).** Both scans show the same departure from a quadratic
  aberration surface: about −5.3 µm of apparent C30, plus ~300 Å each of C21 and C23. It could
  equally be radial distortion of the pattern on the detector. Parallax corrects quadratic
  aberrations alone (Varnavides et al., *Microsc. Microanal.* 32(2), ozaf139, eq. 9), so none of
  this goes into `run.py`.
- **Independent check.** quantem's residual loss (`prlx` and SSB, whole disk) puts C10 at −300 Å
  (25 Å grid) and θ at 5° (2° grid). It is flat along the C10–C30 trade-off, so it doesn't
  confirm the C30.

## Reconstructions

These were made with `run.py --ingest python` on the Mac from the recordings, with the
calibrated flags, at upsampling 1/2/4/8, with the frame mean kept or subtracted, and with DC
correction on or off. All 30 runs report `COMPLETE`.

- **Against quantem.** The same geometry in quantem: `DirectPtychography` and
  `DirectPtychographyMontage`, kernels `prlx` (with and without the phase flip), SSB and OBF, at
  ×1/×2/×4. The data were column-flipped and the origin forced to the measured centre. Pearson r
  against our phase-flipped image (`analysis/figures/realtime_stem_vs_quantem_binned.jpg`):

  | quantem | binned ×1 / ×2 / ×4 | unbinned ×1 / ×2 / ×4 |
  |---|---|---|
  | montage `prlx` + flip | 0.99 / 0.98 / 0.95 | 1.00 / 1.00 / 0.99 |
  | direct `prlx` + flip | 0.93 / 0.97 / 0.93 | 0.92 / 0.98 / 0.98 |
  | SSB | 0.93 / 0.95 / 0.89 | 0.93 / 0.98 / 0.95 |
  | OBF | 0.89 / 0.93 / 0.87 | 0.89 / 0.95 / 0.93 |

  The montage deposits on whole pixels, as we do, so it agrees most closely. At this defocus and
  dose, SSB and OBF look much like phase-flipped parallax.
- **Frame mean** (`frame_mean.jpg`).
  - The per-position mean over the bright-field disk is a clean incoherent bright-field image of
    the particles.
  - Across a particle's wall the bright-field intensity drops about 4%, and the default
    reconstruction (mean kept) follows it.
  - `--subtract-frame-mean` removes 97–98% of the image's power below 1/(2α|C10|) ≈ 1/19 Å⁻¹:
    the amplitude contrast, and with it the scan-line streaks.
- **DC correction** (each bright-field pixel's scan mean removed at readout; `midscan_binned_x4.jpg`,
  `frame_mean_dc_upsampling_binned.jpg`).
  - On a complete scan it removes a constant at ×1 and ×2, and at ×4–×8 the comb of the scan
    lattice at multiples of 1/1.98 Å⁻¹ (12–30% of the image rms).
  - Mid-scan it removes the mean bright-field pedestal, which otherwise steps up about 9× the
    image rms at the scan front.
- **Upsampling.**
  - At ×1 the scan's Nyquist frequency (0.253 Å⁻¹) is below the first zero of sin χ (0.355 Å⁻¹ at
    −316 Å), so the phase-flipped image is just the inverted parallax image, and the lattice
    aliases.
  - At ×2–×8 lattice spots appear at 3.11–3.24 Å (bixbyite 222: 3.12 Å) and at 2.02–2.11 Å.
- **Binned against unbinned** (`binned_vs_unbinned.jpg`). Their FRC is about 0.15 out to
  ~0.1 Å⁻¹, with a lattice peak at 222 that only appears when upsampled.

## The live view and continuous scanning

- **Mid-scan** (`midscan_binned_x4.jpg`). The live channels at 25/50/75/100% of the scan, with
  and without DC correction, phase-flipped, and with `--normalize`.
- **Continuous modes** (`continuous_modes.jpg`; `tools/repeat_dstream.py` +
  `scripts/compare_accumulate.py`). The binned scan was looped 4× with a (3, 4) px drift per
  pass. Halfway through a pass:
  - `latest` and `swap` correlate 0.51 / 0.49 with the pass in progress and 0.55 / 0.56 with the
    one before;
  - `reset` correlates 0.69 with the pass in progress, with half the image empty;
  - `all` blurs the passes together.

  Every mode but `all` equals each pass at its end. `swap` needs one extra image (1 MB at ×2)
  where `latest` stores every frame's bright-field pixels (649 MB).
- **Focus-walk demo.** `tools/refocus_dstream.py` rewrites each pass as if the microscope's
  defocus walked −500 → −450 → −400 → −350 → −300 Å and settled at −316 Å. Replayed in a loop into
  the browser viewer, the image sharpens behind the scan cursor. Checked: a pass written at
  −500 Å measures −501 Å. The commands are in [`docs/WORKFLOWS.md`](../docs/WORKFLOWS.md#a-focus-walk-demo).

## Reproducing

With the recordings in this folder, the repository's environment
(`uv pip install -e ".[local,dev,oracles,viz,replay]"`) and quantem's
(`../quantem-upstream/.venv`):

```bash
out_gd2o3_30mrad/analysis/scripts/reproduce.sh      # ~45 min on an M4 Pro, mostly quantem
```

It runs, in order:
1. the calibration;
2. the 30 reconstructions;
3. the quantem export, reconstructions and residual loss;
4. the continuous-mode comparison;
5. the figures.

Each step is also a script of its own; see [`analysis/README.md`](analysis/README.md).
