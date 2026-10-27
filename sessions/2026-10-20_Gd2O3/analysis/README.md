# 2026-10-20 Gd₂O₃: the session's own scripts and figures

The standard analysis (calibration, reconstructions, `figures/overview.jpg`,
`figures/upsampling_*.jpg`, `summary.json`) is `scripts/session_report.py`'s, the same for every
session ([`sessions/README.md`](../../README.md)). These scripts add what is particular to this
session. The findings are in [`../README.md`](../README.md).

## Scripts

`reproduce.sh` runs everything in order. Unless the table says otherwise, a script uses the
repository's environment (`.venv`) and can be run from anywhere. Paths are in `paths.py`.

| script | makes | time |
|---|---|---|
| (step 1 of `reproduce.sh`) `scripts/session_report.py` | `calibration/`, `recon/<id>/U<u>[_fit\|_live\|_sub\|_nodc]` (the variants from `session.json`), `figures/overview.jpg`, `figures/upsampling_*.jpg`, `summary.json` | 2 min |
| `export_for_quantem.py` | `recon/quantem_inputs/`: both recordings as 4D arrays, column-flipped to quantem's layout (the unbinned one cropped to 128²) | 1 min |
| `quantem_recons.py binned\|unbinned` (quantem's environment) | `recon/quantem/`: `DirectPtychography` and `DirectPtychographyMontage`, kernels `prlx` (± flip), SSB and OBF, ×1/×2/×4, with the calibrated geometry (C10 refined, as `run.py` gets it) | 3 / 15 min |
| `obf_vs_quantem.py binned\|unbinned [U]` (quantem's environment) | `recon/obf/<id>_obf_U<u>.npy`: live OBF (`processing/obf`) of the whole scan; prints it against quantem's correlation route on identical inputs (2e-6 relative at ×2) and against quantem's own OBF from `quantem_recons.py` (0.9988 / 0.9998 interior correlation) | 1 / 3 min |
| `quantem_loss.py` (quantem's environment) | prints quantem's residual loss along C10 and θ, over (C10, C30), and at the live, calibrated, higher-order and twin settings (about the shift fit) | 15 min |
| (step 3 of `reproduce.sh`) `tools/repeat_dstream.py` + `scripts/compare_accumulate.py` | `figures/full/continuous_modes.png` | 1 min |
| `figures.py` | `figures/full/realtime_stem_vs_quantem_<id>.png`, `frame_mean_dc_upsampling_<id>.png`, `binned_vs_unbinned.png`; `analysis/summary.json` (correlations, drift) | 20 s |
| `frame_mean.py` | `figures/full/frame_mean.png` | 10 s |
| `midscan.py` | `figures/full/midscan_<id>_x<u>.png` | 10 s |
| `publish_figures.py` | `figures/<name>.jpg`: the downscaled copies kept in git | 5 s |

## Figures kept in git (`figures/`)

| figure | shows |
|---|---|
| `overview.jpg` | (standard) Each recording at the live settings, the support focus and the lattice focus, ×4 phase-flipped. Also its spectrum with the bixbyite rings, and the calibration along C10. |
| `upsampling_binned.jpg`, `upsampling_unbinned.jpg` | (standard) ×1/×2/×4/×8 at the calibrated C10, a crop where the lattice is strongest, and the spectra. |
| `realtime_stem_vs_quantem_binned.jpg` | A 12.6 nm region at ×1/×2/×4: our parallax and phase-flipped images, and quantem's direct and montage `prlx` (± flip), SSB and OBF. With power spectra and bixbyite rings. |
| `frame_mean_dc_upsampling_binned.jpg` | Mean kept against `--subtract-frame-mean`, and what DC correction removes, at ×1/2/4/8. Spectra on a common ±0.75 Å⁻¹ axis. |
| `binned_vs_unbinned.jpg` | Both scans at ×4 (registered), their spectra, and their FRC at ×1/2/4. |
| `midscan_binned_x4.jpg` | The live view at 25/50/75/100% of the binned scan at ×4: parallax with DC correction on and off, phase-flipped, and with `--normalize`. |
| `frame_mean.jpg` | What `--subtract-frame-mean` removes, for both scans: the frame-mean image, the difference, power spectra, and a profile across a particle. |
| `continuous_modes.jpg` | `latest`, `all`, `reset`, `swap` and `swap β=0.5` live images at 25/50/75/100% of passes 1–3 of a 4-pass loop with drift. Each is labelled with its correlation to the pass in progress and to the one before. |

The calibration plots are `calibration/<id>.png`. The full-resolution figures (`figures/full/`,
up to ~10 MB each) are not in git.
