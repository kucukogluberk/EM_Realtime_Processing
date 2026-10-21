# Analysis of the Gd₂O₃ recordings: scripts and figures

What each script does and makes, and the figures kept in git. The findings are in
[`../README.md`](../README.md).

## Scripts (`scripts/`)

`reproduce.sh` runs them all in order. Unless the table says otherwise, a script uses the
repository's environment (`.venv`) and can be run from anywhere. Paths are in `scripts/paths.py`.

| script | makes | time |
|---|---|---|
| (step 1 of `reproduce.sh`) `python -m tools.calibrate_parallax … --out analysis/cal_<mode>` | `cal_{binned,unbinned}.flags` (the `run.py` flags), `.json` (everything measured), `.png` | 30 s |
| `run_variants.sh` | `../recon/<mode>_U<u>[_sub\|_nodc]/`: both recordings at ×1/2/4/8, with the frame mean kept, subtracted (`_sub`) or DC correction off (`_nodc`). Set `DEVICE=cuda` or `cpu` off a Mac. | 2 min |
| `export_for_quantem.py` | `../quantem_inputs/`: both recordings as 4D arrays, column-flipped to quantem's layout (the unbinned one cropped to 128²) | 1 min |
| `quantem_recons.py binned\|unbinned` (quantem's environment) | `../recon/quantem/`: `DirectPtychography` and `DirectPtychographyMontage`, kernels `prlx` (± flip), SSB and OBF, ×1/×2/×4, with the calibrated geometry | 10 / 25 min |
| `quantem_loss.py` (quantem's environment) | prints quantem's residual loss along C10 and θ, over (C10, C30), and at the live, calibrated, higher-order and twin settings | 15 min |
| `figures.py` | `overview.png`, `B1_methods_*.png`, `B2_variants_*.png`, `B3_binned_vs_unbinned.png`, `B_summary.json` | 20 s |
| `frame_mean.py` | `B5_frame_mean.png` | 10 s |
| `midscan.py` | `B4_midscan_*.png` | 10 s |
| (step 4 of `reproduce.sh`) `tools/repeat_dstream.py` + `scripts/compare_accumulate.py` | `C_compare_modes.png` | 1 min |
| `publish_figures.py` | `figures/`: the downscaled copies kept in git | 5 s |

The full-resolution figures (`analysis/*.png`, up to ~10 MB each) are not in git.

## Figures kept in git (`figures/`)

| figure | shows |
|---|---|
| `calibration_binned.png` | The calibration of the binned scan. Panels: mean diffraction pattern and disk; the measured and fitted vBF shifts; their residual; the radial shift per tilt (the apparent C30); the variance loss along C10, θ and for the other handedness; the vBF sum before and after shifting. |
| `overview.png` | Both recordings: parallax and phase-flipped at ×1, phase-flipped at ×4. |
| `realtime_stem_vs_quantem_binned.jpg` | A 12.6 nm region at ×1/×2/×4 for our parallax and phase-flipped images and quantem's direct and montage `prlx` (± flip), SSB and OBF, with power spectra and bixbyite rings (211/222/400/440/622). |
| `frame_mean_dc_upsampling_binned.jpg` | Mean kept against `--subtract-frame-mean`, and what DC correction removes, at ×1/2/4/8, with spectra on a common ±0.75 Å⁻¹ axis. |
| `binned_vs_unbinned.jpg` | Both scans at ×4 (registered), their spectra, and their FRC at ×1/2/4. |
| `midscan_binned_x4.jpg` | The live view at 25/50/75/100% of the binned scan at ×4: parallax with DC correction on and off, phase-flipped, and with `--normalize`. |
| `frame_mean.jpg` | What `--subtract-frame-mean` removes, for both scans: the frame-mean image, the difference, power spectra, and a profile across a particle. |
| `continuous_modes.jpg` | `latest`, `all`, `reset`, `swap` and `swap β=0.5` live images at 25/50/75/100% of passes 1–3 of a 4-pass loop with drift. Each is labelled with its correlation to the pass in progress and to the one before. |
