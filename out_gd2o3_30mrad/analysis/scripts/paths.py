"""Where everything of the 2026-10-20 Gd2O3 session lives, relative to this folder (no imports
from the repo, so the quantem scripts can use it from quantem's own environment)."""
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ANALYSIS = SCRIPTS.parent                       # out_gd2o3_30mrad/analysis
EXP = ANALYSIS.parent                           # out_gd2o3_30mrad
REPO = EXP.parent
RECON = EXP / "recon"                           # run.py results (not in git)
QUANTEM = RECON / "quantem"                     # quantem results (not in git)
INPUTS = EXP / "quantem_inputs"                 # 4D arrays exported for quantem (not in git)
FIGURES = ANALYSIS / "figures"                  # the downscaled figures kept in git

STEM = "Gd2O3_50.6nmFOV_256x256_30mrad_-30nm_"
RECORDINGS = {"binned": EXP / f"{STEM}binned_10us.dstream",
              "unbinned": EXP / f"{STEM}unbinned.dstream"}
HAADF = {"binned": EXP / f"{STEM}binned_10us_HAADF.npy",
         "unbinned": EXP / f"{STEM}unbinned_HAADF.npy"}
STEP = 1.9766                                   # Å, 50.6 nm over 256 px


def cal(mode: str) -> Path:
    """The calibration of a recording (tools/calibrate_parallax.py --out analysis/cal_<mode>)."""
    return ANALYSIS / f"cal_{mode}.json"


def flags(mode: str) -> list[str]:
    return (ANALYSIS / f"cal_{mode}.flags").read_text().split()
