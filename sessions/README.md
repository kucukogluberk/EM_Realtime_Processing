# Sessions: recordings at the microscope, analysed the same way

Each folder here is one session at the microscope: its recordings, its calibrations, its
reconstructions and what we learned from them. Every session has the same layout and gets the
same standard analysis, so sessions can be compared directly.

| session | specimen | recordings | the focus |
|---|---|---|---|
| [2026-10-20_Gd2O3](2026-10-20_Gd2O3/README.md) | Gd₂O₃ nanoparticles on carbon, 30 mrad | 256² binned and unbinned | the first live session; calibration, checked against quantem; continuous modes |
| [2026-10-23_Au-grating](2026-10-23_Au-grating/README.md) | Au islands on a carbon grating replica | 128² binned, disk centred and not | the new web viewer, A1 tuning, upsampling; the islands focus 150 Å from the support |

The recordings themselves are not in git. Ask Georgios Varnavides or Berk Kucukoglu for them.

## The layout

```
sessions/<YYYY-MM-DD>_<specimen>/
  README.md        the data, the calibration, what the reconstructions show, how to reproduce
  session.json     the recordings: file, acquisition values, and the flags used live
  summary.json     the standard numbers (scripts/session_report.py)
  .gitignore       keeps data/, recon/ and figures/full/ out of git
  data/            as recorded: .dstream, screen recordings, viewer exports (not in git)
  calibration/     <id>.flags (run.py flags), <id>.json (everything measured), <id>.png
  recon/           <id>/U<u>/ … run.py results (not in git)
  figures/         the figures kept in git (downscaled); figures/full/ the originals (not in git)
  analysis/        the session's own scripts, with a README and reproduce.sh
```

`<id>` is a short name for a recording (`binned`, `centered`, …), set in `session.json`.

## The standard analysis: `scripts/session_report.py`

```bash
python scripts/session_report.py sessions/<session>          # calibrate, recon, figures
python scripts/session_report.py sessions/<session> --steps figures
```

For each recording:

1. **`calibration/<id>.*`** come from `tools/calibrate_parallax.py`: the disk, the column flip,
   the rotation, C10 and C12.
   - C10 is refined on the lattice. The shifts measure the support film's focus; `--defocus` is
     where the particles' lattice is sharpest.
2. **`recon/<id>/U1, U2, U4, U8`** are the reconstructions at ×1/2/4/8 with those flags.
3. **`recon/<id>/U4_fit`** is ×4 at the shift fit's C10, the support's focus.
4. **`recon/<id>/U4_live`** is ×4 with the flags used live, from `session.json`.
5. **`recon/<id>/U<u>_<variant>`** are any extra variants in `session.json` (e.g. `--subtract-frame-mean`).

For the session:

- **`figures/overview.jpg`:** each recording at the live settings, the support focus and the
  lattice focus. It also shows the power spectrum, with the specimen's reflections ringed, and
  the calibration along C10: the variance loss (support) and the lattice prominence.
- **`figures/upsampling_<id>.jpg`:** ×1/×2/×4/×8, with a crop where the lattice is strongest
  and the spectra.
- **`summary.json`:** the calibration and the status of every reconstruction.

## Adding a session

1. Make `sessions/<date>_<specimen>/` and copy this layout's `.gitignore` from another session.
   Put the recordings in `data/`, keeping the operator's subfolders.
2. Write `session.json`. Copy another session's and change the files, the acquisition values
   (`calibrate`: what `tools/calibrate_parallax.py` needs), the live flags if known, and the
   reflections to ring.
   - Don't carry acquisition values over from another session. On 2026-10-23 the disk was 6%
     larger than on 2026-10-20 (see that session's README).
3. Run `python scripts/session_report.py sessions/<session>`.
4. Put the session's own analyses in `analysis/`, writing to `figures/full/` and publishing the
   ones worth keeping to `figures/`. Then write the README.
