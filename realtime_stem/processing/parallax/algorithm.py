"""ParallaxAlgorithm — wire parallax / tcBF into the generic runner.

This is the whole "how to add a method" story in one file: declare the CLI flags that shape the
geometry, build the accumulator from them, and save the result. The runner (`apps/run.py`) and the
ingest graph never mention parallax — they see only the `Algorithm` / `Accumulator` contracts.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from realtime_stem.processing.base import Accumulator, Algorithm
from realtime_stem.processing.parallax.config import (
    add_tuning_args,
    config_from_args,
    join_negative_floats,
)
from realtime_stem.processing.parallax.engine import (
    ParallaxAccumulator,
    preprocess_geometry,
)


class ParallaxAlgorithm(Algorithm):
    name = "parallax"

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        # Geometry presets first, then the shared physics/tuning knobs.
        parser.add_argument(
            "--detector", type=int, nargs=2, default=None, metavar=("HY", "HX"),
            help="Detector geometry (default: config, 192x192; --binned preset: 96x96). Changing "
                 "the detector changes the BF disk — set --angular-sampling to match the mode.")
        parser.add_argument(
            "--binned", action="store_true",
            help="ARINA 2×2-binned preset: 96×96 detector, doubled angular sampling.")
        add_tuning_args(parser)   # --defocus/--rotation/--upsampling/physics/--scan/--center

    def preprocess_argv(self, argv: list[str]) -> list[str]:
        # `--defocus -1.5e4` etc.: scientific-notation negatives read as flags otherwise.
        return join_negative_floats(argv)

    def build(self, args: argparse.Namespace) -> Accumulator:
        cfg = config_from_args(args)
        if args.detector is not None:
            cfg = replace(cfg, detector_shape=(int(args.detector[0]), int(args.detector[1])))

        geo = preprocess_geometry(
            det_shape=cfg.detector_shape, scan_shape=cfg.scan_shape,
            scan_sampling=cfg.scan_sampling, energy=cfg.energy,
            semiangle_cutoff=cfg.semiangle_cutoff, angular_sampling=cfg.angular_sampling,
            aberration_coefs=cfg.aberration_coefs, rotation_angle=cfg.rotation_angle,
            upsampling_factor=cfg.upsampling_factor, probe_center=cfg.probe_center,
        )

        rot_deg = 0.0 if cfg.rotation_angle is None else float(np.rad2deg(cfg.rotation_angle))
        center_str = ("frame center" if cfg.probe_center is None
                      else f"({cfg.probe_center[0]:.4g}, {cfg.probe_center[1]:.4g})")
        print(f"[parallax] BF pixels = {geo.n_bf}, "
              f"C10 = {cfg.aberration_coefs.get('C10', 0.0):.4g} Å, rotation = {rot_deg:.4g}°, "
              f"max |shift| = {int(np.abs(geo.shifts).max())} px, upsampling = {cfg.upsampling_factor}, "
              f"angular sampling = {cfg.angular_sampling[0]:.4g} mrad/px, disk center = {center_str}",
              flush=True)

        return ParallaxAccumulator(geo, device="cuda")

    def save(self, acc: Accumulator, out_dir: Path, tag: str = "") -> None:
        recon = acc.result()
        np.save(out_dir / "parallax_recon.npy", recon)                   # always the latest scan
        np.save(out_dir / "parallax_shifts.npy", acc.geometry.shifts)   # parallax-specific extra
        saved = "parallax_recon.npy + parallax_shifts.npy"
        if tag:
            np.save(out_dir / f"parallax_recon_{tag}.npy", recon)
            saved += f" + parallax_recon_{tag}.npy"
        print(f"[parallax] saved {saved} to {out_dir}", flush=True)
