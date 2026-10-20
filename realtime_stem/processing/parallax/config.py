"""
Configuration for the real-time parallax / tcBF pipeline.

All tunable parameters live here as plain data so they can be edited without touching logic. On
top of the detector/scan geometry it carries the parallax-specific **acquisition physics** the
geometry preprocessing needs: beam energy, convergence semiangle, detector angular sampling, scan
sampling, and the aberration coefficients (defocus C10, optional astigmatism C12/phi12) that set
the parallax shifts.

Parallax needs a *defocus* to compute shifts: the microscope's reported defocus is the starting
point (override live with `--defocus`); the shift field itself is observable, so a future step can
refine C10 from the data.

The values here are plausible ARINA-scale examples, not measured constants — set them on the CLI
(`add_tuning_args`, below) to match your acquisition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ParallaxConfig:
    # ── Detector geometry ─────────────────────────────────────────────────────
    # Swap the active line, or override with --detector / --binned.
    detector_shape: tuple[int, int] = (192, 192)    # ARINA full resolution (unbinned)
    # detector_shape: tuple[int, int] = (96, 96)    # ARINA binned ×2

    # ── Scan geometry ─────────────────────────────────────────────────────────
    # Frames arrive in raster order and are placed by message id: (id // Nx, id % Nx).
    scan_shape: tuple[int, int] = (1024, 1024)      # override with --scan NY NX

    # ── Parallax acquisition physics (for preprocess_geometry) ────────────────
    energy: float = 300e3                               # beam energy, eV
    semiangle_cutoff: float = 4.0                       # probe convergence semiangle, mrad
    angular_sampling: tuple[float, float] = (0.5, 0.5)  # detector pixel size, mrad/px
    scan_sampling: tuple[float, float] = (0.5, 0.5)     # scan step, Å
    # Aberrations in Å / Å / rad. C10 = defocus (the dominant term; negative = underfocus in the
    # reference's sign convention). Override C10 live with --defocus.
    aberration_coefs: dict = field(default_factory=lambda: {"C10": -1.5e4})
    rotation_angle: float | None = None                 # scan/detector rotation, rad
    upsampling_factor: int = 1                          # output-grid upsampling (finer shifts)
    # BF-disk center on the detector, (row, col) px — py4DSTEM get_probe_size's
    # (Center_X, Center_Y). None → frame center. Real beams sit off-center; override with --center.
    probe_center: tuple[float, float] | None = None

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def n_scan_y(self) -> int:
        return self.scan_shape[0]

    @property
    def n_scan_x(self) -> int:
        return self.scan_shape[1]

    @property
    def num_positions(self) -> int:
        return self.scan_shape[0] * self.scan_shape[1]


# Default instance imported by the rest of the pipeline. Construct a different
# ParallaxConfig(...) (e.g. in tests or for another scan) to override.
CONFIG = ParallaxConfig()


# ── shared CLI tuning knobs ───────────────────────────────────────────────────
# The tuning-knob flag group and its unit conversions live HERE (not in run.py) so the parallax
# algorithm owns its CLI in one place — the same flags, with the same units, whether you tune a
# defocus offline against a replay or drive it live. CLI angles are degrees (the microscope readout).

def add_tuning_args(p) -> None:
    """Attach the shared parallax tuning flags to an argparse parser."""
    p.add_argument("--defocus", type=float, default=None, metavar="C10_ANG",
                   help="Override the defocus C10 (Å) used to build the parallax shifts.")
    p.add_argument("--rotation", type=float, default=None, metavar="DEG",
                   help="Override the scan/detector rotation (degrees).")
    p.add_argument("--c12", type=float, default=None, metavar="C12_ANG",
                   help="Override the astigmatism magnitude C12 (Å).")
    p.add_argument("--phi12", type=float, default=None, metavar="DEG",
                   help="Override the astigmatism angle phi12 (degrees).")
    p.add_argument("--upsampling", type=int, default=None, metavar="U",
                   help="Override config.upsampling_factor (finer output grid → finer shifts).")
    p.add_argument("--energy", type=float, default=None, metavar="EV",
                   help="Override the beam energy (eV).")
    p.add_argument("--semiangle", type=float, default=None, metavar="MRAD",
                   help="Override the probe convergence semiangle cutoff (mrad).")
    p.add_argument("--angular-sampling", type=float, default=None, metavar="MRAD_PX",
                   help="Override the detector angular sampling (mrad/px, isotropic). "
                        "Give the value for the mode actually running (wins over --binned doubling).")
    p.add_argument("--scan-sampling", type=float, default=None, metavar="ANG_PX",
                   help="Override the scan step (Å, isotropic).")
    p.add_argument("--center", type=float, nargs=2, default=None, metavar=("ROW", "COL"),
                   help="BF-disk center on the detector (px), py4DSTEM's Center_X Center_Y "
                        "from get_probe_size. Default: the frame center.")
    p.add_argument("--scan", type=int, nargs=2, default=None, metavar=("NY", "NX"),
                   help="Override the scan grid (must match what the microscope drives).")


# Float-valued tuning flags whose natural values are often negative (e.g. underfocus C10).
_TUNING_FLOAT_FLAGS = {"--defocus", "--rotation", "--c12", "--phi12", "--energy",
                       "--semiangle", "--angular-sampling", "--scan-sampling"}


def join_negative_floats(argv: list[str]) -> list[str]:
    """Rewrite `--defocus -1.5e4` to `--defocus=-1.5e4` — argparse only auto-recognizes plain
    negative numbers (`-15000`), not scientific notation, so `-1.5e4` parses as a flag."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in _TUNING_FLOAT_FLAGS and i + 1 < len(argv) and argv[i + 1].startswith("-"):
            try:
                float(argv[i + 1])
            except ValueError:
                pass
            else:
                out.append(f"{a}={argv[i + 1]}")
                i += 2
                continue
        out.append(a)
        i += 1
    return out


def config_from_args(args, cfg: ParallaxConfig | None = None) -> ParallaxConfig:
    """Apply the CLI tuning knobs to the config — the one place units are converted.

    CLI angles are degrees (what the microscope UI shows) and become radians here; a single
    sampling value becomes an isotropic (y, x) tuple. `--binned` doubles the config-default angular
    sampling, but an explicit `--angular-sampling` describes the actual running mode and wins.
    """
    if cfg is None:
        cfg = CONFIG
    overrides = {}
    if getattr(args, "binned", False):
        # 2×2 binning halves the detector and doubles the per-pixel angular sampling.
        overrides.update(detector_shape=(96, 96),
                         angular_sampling=(cfg.angular_sampling[0] * 2, cfg.angular_sampling[1] * 2))

    aberr = dict(cfg.aberration_coefs)
    if args.defocus is not None:
        aberr["C10"] = args.defocus
    if args.c12 is not None:
        aberr["C12"] = args.c12
    if args.phi12 is not None:
        aberr["phi12"] = math.radians(args.phi12)
    if aberr != cfg.aberration_coefs:
        overrides["aberration_coefs"] = aberr

    if args.rotation is not None:
        overrides["rotation_angle"] = math.radians(args.rotation)
    if args.upsampling is not None:
        overrides["upsampling_factor"] = args.upsampling
    if args.energy is not None:
        overrides["energy"] = args.energy
    if args.semiangle is not None:
        overrides["semiangle_cutoff"] = args.semiangle
    if args.angular_sampling is not None:
        overrides["angular_sampling"] = (args.angular_sampling, args.angular_sampling)
    if args.scan_sampling is not None:
        overrides["scan_sampling"] = (args.scan_sampling, args.scan_sampling)
    if args.center is not None:
        overrides["probe_center"] = (args.center[0], args.center[1])
    if args.scan is not None:
        overrides["scan_shape"] = (int(args.scan[0]), int(args.scan[1]))

    return replace(cfg, **overrides) if overrides else cfg   # frozen dataclass → replace()
