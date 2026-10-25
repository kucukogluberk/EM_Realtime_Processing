# Gd₂O₃ nanoparticles, 2026-10-20

The first live session of realtime-stem on the ARINA, and what we learned from its recordings
afterwards:
- the measured geometry, with C10 refined on the particles' lattice;
- offline reconstructions, checked against quantem;
- the frame mean and the DC correction;
- the live view part-way through a scan;
- the continuous-scanning modes;
- a focus-walk demo.

The recordings are **not in git** (0.7 GB; ask Georgios Varnavides or Berk Kucukoglu). The
layout is [`sessions/README.md`](../README.md)'s; [`analysis/README.md`](analysis/README.md)
lists this session's own scripts and figures. For the tools themselves (calibration, looping a
recording, the browser viewer), see [`docs/WORKFLOWS.md`](../../docs/WORKFLOWS.md).

## The data

| | |
|---|---|
| microscope | 200 keV, convergence semiangle 30 mrad, ARINA D030056 |
| `data/…_binned_10us.dstream` | 256² scan over 50.6 nm (1.9766 Å step), 65,536 frames of 96 × 96 (2×2 binned) uint16, 10 µs; ~10k counts per frame; 1.066 mrad/px |
| `data/…_unbinned.dstream` | the same field, 192 × 192 uint16, 34 µs; ~29k counts per frame; 0.533 mrad/px; recorded 11 min before the binned scan |
| `data/…_HAADF.npy` (×2) | 256² HAADF images of the same scans; at these dwell times mostly shot noise |
| `data/recon_20261020_*.mkv` (×6) | screen recordings of the live reconstructions (`--accumulate latest --continuous`); the scan, upsampling and dwell of each are in its name |
| live settings | `--defocus=-300` (the microscope's "−30 nm"), `--rotation=3.75 --detector-layout rows,-cols`, the disk centre at its default, the frame mean kept |

## Calibration

`tools/calibrate_parallax.py` measures the shift of each small group of bright-field pixels and
fits the engine's shift model, M = (C10·I + C12·Q(φ12))·R(θ)·diag(1, ±1).
- **Edge margin:** shifts are measured below 0.18 Å⁻¹, from pixels clear of the disk edge by
  that frequency. Near the edge only one sideband is transferred, which moves the correlation
  peak outwards.
- **Lattice refinement:** C10 is then refined on the lattice.
- **Outputs:** `calibration/<id>.flags` hold the `run.py` flags; `calibration/<id>.png` is the
  diagnostic plot.

| | live | binned | unbinned |
|---|---|---|---|
| disk centre (row, col), raw frame | default (48, 47) / (96, 95) | (47.86, 48.41) | (96.24, 97.32) |
| disk radius | | 28.33 px = 30.2 mrad | 56.57 px = 30.2 mrad |
| columns | flipped (`rows,-cols`) | flipped | flipped |
| rotation | 3.75° | 4.78 ± 0.09° | 4.77 ± 0.08° |
| C10, shift fit (the support) | | −315.6 ± 0.9 Å | −296.1 ± 0.8 Å |
| **C10, refined on the lattice** (`--defocus`) | −300 Å | **−281 Å** (1.6× the lattice prominence) | **−296 Å** (the fit's) |
| C12 | — | 16.7 Å at −7.1° | 12.9 Å at −8.1° |

- **The particles' lattice.**
  - In the binned scan the bixbyite spots are most prominent 35 Å (one depth of field,
    λ/α² = 28 Å) closer to focus than the support. That's the particles sitting above the carbon.
  - In the unbinned scan they peak at the fit itself.
  - Both effects are far weaker than on the 2026-10-23 Au grating, where the gap was 150 Å.
- **Twin and sign.** The shifts can't tell (C10, θ) from (−C10, θ + 180°). Taking the rotation
  near the live 3.75°, C10 is negative, i.e. underfocus in abTEM's convention, as "−30 nm"
  says.
- **Drift.** The 20 Å between the two scans' fits is presumably focus drift over the 11 minutes
  between them. The specimen also moved (+4.9, −6.4) Å.
- **Defective pixels.** All lie outside the disk: three pixels flagged 65535 unbinned, binned
  into two hot pixels. Nothing needs masking.
- **What the live run had wrong:** the rotation by 1°, and the default centre by 1.5–2.5 px in
  columns. The centre error mostly moves the whole image.
- **Beyond quadratic (a diagnostic only).**
  - Both scans show the same departure from a quadratic aberration surface: about −5.3 µm of
    apparent C30, plus ~300 Å each of C21 and C23. It could equally be radial distortion of the
    pattern on the detector.
  - Parallax corrects quadratic aberrations alone (Varnavides et al., *Microsc. Microanal.*
    32(2), ozaf139, eq. 9), so none of this goes into `run.py`.
- **Independent check.**
  - quantem's residual loss (whole disk) puts C10 at −325 Å for `prlx` and −300 Å for SSB
    (25 Å grid), and θ at 5° (2° grid). Like the shift fit, it measures the low-frequency
    contrast.
  - It is flat along the C10–C30 trade-off, so it doesn't confirm the C30.
- **Angular sampling × scan step.** The CTF zeros of the support-focus reconstruction sit 3–5%
  below where the calibration puts them (r = 0.95–0.96; the 2026-10-23 session's
  `analysis/ctf_zeros.py`). The fitted C10 itself carries about 3%, so the 30 mrad and
  1.066 mrad/px stand.

## Reconstructions

`figures/overview.jpg` shows each recording at the live settings, the support focus and the
lattice focus, its spectrum, and the calibration along C10. `figures/upsampling_<id>.jpg`
shows ×1/×2/×4/×8.

All 28 standard reconstructions report `COMPLETE`: ×1/2/4/8, the support focus, the live flags,
and ×1–8 with the frame mean subtracted and with DC correction off.

- **Against quantem** (`figures/realtime_stem_vs_quantem_binned.jpg`).
  - quantem was run with the same geometry, C10 refined: `DirectPtychography` and
    `DirectPtychographyMontage`, with kernels `prlx` (with and without the phase flip), SSB and
    OBF, at ×1/×2/×4.
  - The data were column-flipped and the origin forced to the measured centre.
  - Pearson r against our phase-flipped image:

  | quantem | binned ×1 / ×2 / ×4 | unbinned ×1 / ×2 / ×4 |
  |---|---|---|
  | montage `prlx` + flip | 0.99 / 0.99 / 0.96 | 1.00 / 1.00 / 0.99 |
  | direct `prlx` + flip | 0.91 / 0.96 / 0.92 | 0.92 / 0.98 / 0.98 |
  | SSB | 0.92 / 0.95 / 0.89 | 0.93 / 0.98 / 0.95 |
  | OBF | 0.87 / 0.93 / 0.86 | 0.89 / 0.95 / 0.93 |

  - The montage deposits on whole pixels, as we do, so it agrees most closely.
  - At this defocus and dose, SSB and OBF look much like phase-flipped parallax.
- **Frame mean** (`figures/frame_mean.jpg`).
  - The per-position mean over the bright-field disk is a clean incoherent bright-field image of
    the particles.
  - Across a particle's wall the bright-field intensity drops about 4%, and the default
    reconstruction (mean kept) follows it.
  - `--subtract-frame-mean` removes 97–98% of the image's power below 1/(2α|C10|) ≈ 1/17 Å⁻¹:
    the amplitude contrast, and with it the scan-line streaks.
- **DC correction** (each bright-field pixel's scan mean removed at readout;
  `figures/midscan_binned_x4.jpg`, `figures/frame_mean_dc_upsampling_binned.jpg`).
  - On a complete scan it removes a constant at ×1 and ×2, and at ×4–×8 the comb of the scan
    lattice at multiples of 1/1.98 Å⁻¹ (12–30% of the image rms).
  - Mid-scan it removes the mean bright-field pedestal, which otherwise steps up about 9× the
    image rms at the scan front.
- **Upsampling.**
  - At ×1 the scan's Nyquist frequency (0.253 Å⁻¹) is below the first zero of sin χ, so the
    phase-flipped image is just the inverted parallax image, and the lattice aliases.
  - At ×2–×8 bixbyite spots appear (222 and others).
- **Binned against unbinned** (`figures/binned_vs_unbinned.jpg`). Their FRC is about 0.15 out to
  ~0.1 Å⁻¹, with a lattice peak at 222 that only appears when upsampled.

## The live session

- **Mid-scan** (`figures/midscan_binned_x4.jpg`). The live channels at 25/50/75/100% of the
  scan, with and without DC correction, phase-flipped, and with `--normalize`.
- **Continuous modes** (`figures/continuous_modes.jpg`; `tools/repeat_dstream.py` +
  `scripts/compare_accumulate.py`).
  - The binned scan was looped 4× with a (3, 4) px drift per pass.
  - Halfway through a pass, `latest` and `swap` correlate about 0.5 with the pass in progress
    and about 0.55 with the one before.
  - `reset` correlates about 0.7 with the pass in progress, with half the image empty.
  - `all` blurs the passes together.
  - Every mode but `all` equals each pass at its end.
  - `swap` needs one extra image (1 MB at ×2) where `latest` stores every frame's bright-field
    pixels (649 MB).
- **Focus-walk demo.**
  - `tools/refocus_dstream.py` rewrites each pass as if the microscope's defocus walked
    −500 → −450 → −400 → −350 → −300 Å and settled at the recording's own C10.
  - Replayed in a loop into the browser viewer, the image sharpens behind the scan cursor.
  - Checked: a pass written at −500 Å measures −501 Å.
  - The commands are in [`docs/WORKFLOWS.md`](../../docs/WORKFLOWS.md#a-focus-walk-demo).

## Reproducing

With the recordings in `data/`, the repository's environment
(`uv pip install -e ".[local,dev,oracles,viz,replay]"`) and quantem's (`../quantem-upstream/.venv`):

```bash
sessions/2026-10-20_Gd2O3/analysis/reproduce.sh      # ~50 min on an M4 Pro, mostly quantem
```

It runs:
1. the standard analysis (`scripts/session_report.py`): calibration, the 28 reconstructions, the
   overview and upsampling figures, `summary.json`;
2. quantem's reconstructions and residual loss;
3. the continuous-mode comparison;
4. this session's own figures.

Each step is also a script of its own; see [`analysis/README.md`](analysis/README.md).
