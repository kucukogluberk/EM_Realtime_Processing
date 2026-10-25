# 2026-10-23 Au grating: the session's own scripts and figures

The standard analysis (calibration, reconstructions, `figures/overview.jpg`,
`figures/upsampling_*.jpg`, `summary.json`) is `scripts/session_report.py`'s, the same for every
session ([`sessions/README.md`](../../README.md)). These scripts add what is particular to this
session: the checks behind the calibration's acquisition values. The findings are in
[`../README.md`](../README.md).

## Scripts

`reproduce.sh` runs everything in order. Each script uses the repository's environment (`.venv`)
and can be run from anywhere. Paths are in `paths.py`. Their numbers go to `analysis/summary.json`.

| script | makes | time |
|---|---|---|
| (step 1 of `reproduce.sh`) `scripts/session_report.py` | `calibration/`, `recon/<id>/U<u>[_fit\|_live]`, `figures/overview.jpg`, `figures/upsampling_*.jpg`, `summary.json` | 25 s |
| `disk_size.py` | `figures/full/disk_vs_oct20.png`: the bright-field disk against 2026-10-20. Its 50% edge, the halo, and per 8×8-position tile the centre and radius. | 1 min |
| `scan_step.py` | `figures/full/scan_step.png`: the scan step from the Au 111 / 200 fringes in the live snapshots (`data/Initial_test_A1/snapshot*.npy`) | 5 s |
| `ctf_zeros.py` | `figures/full/ctf_zeros.png`: the CTF zeros of both sessions' support-focus reconstructions against the calibration (angular sampling × scan step) | 10 s |
| `publish_figures.py` | `figures/<name>.jpg`: the downscaled copies kept in git | 2 s |

`disk_size.py` and `ctf_zeros.py` read the 2026-10-20 session too: run its standard analysis
first.

## Figures kept in git (`figures/`)

| figure | shows |
|---|---|
| `overview.jpg` | (standard) Each recording at the presumed live settings, the support focus (−521 / −461 Å) and the lattice focus (−385 / −313 Å), ×4 phase-flipped. Also its spectrum with the Au 111 / 200 / 220 rings, and the calibration along C10: the variance loss (support) against the lattice prominence. |
| `upsampling_centered.jpg`, `upsampling_not_centered.jpg` | (standard) ×1/×2/×4/×8 at the lattice focus, a 5.7 nm crop where the lattice is strongest, and the spectra. |
| `disk_vs_oct20.jpg` | The mean diffraction patterns and their azimuthal profiles. The disk is 29.9 px against 28.1 px on 2026-10-20, the same in every tile. |
| `scan_step.jpg` | The live snapshots and their spectra. 111 at 7.04 px and 200 at 6.07 px (ratio 1.160) give a 1.34 Å step. |
| `ctf_zeros.jpg` | Power along contours of the calibration's χ, for all four recordings. The zeros fall at r = 0.95–0.96 (2026-10-20) and 1.02–1.03 (2026-10-23) of where the calibration puts them. |

The calibration plots are `calibration/<id>.png`. The full-resolution figures (`figures/full/`)
are not in git.
