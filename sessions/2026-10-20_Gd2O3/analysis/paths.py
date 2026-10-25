"""Where everything of the 2026-10-20 Gd2O3 session lives (sessions/README.md has the layout). No
imports from the repo, so the quantem scripts can use it from quantem's own environment."""
import json
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
SESSION = ANALYSIS.parent                        # sessions/2026-10-20_Gd2O3
REPO = SESSION.parents[1]
CALIBRATION = SESSION / "calibration"            # <mode>.flags / .json (scripts/session_report.py)
RECON = SESSION / "recon"                        # <mode>/U<u>[_sub|_nodc|_fit|_live] (not in git)
QUANTEM = RECON / "quantem"                      # quantem results (not in git)
INPUTS = RECON / "quantem_inputs"                # 4D arrays exported for quantem (not in git)
FIGURES = SESSION / "figures"                    # the downscaled figures kept in git
FULL = FIGURES / "full"                          # full-resolution figures (not in git)

_session = json.loads((SESSION / "session.json").read_text())
RECORDINGS = {m: SESSION / r["file"] for m, r in _session["recordings"].items()}
HAADF = {m: p.with_name(p.name.replace(".dstream", "_HAADF.npy")) for m, p in RECORDINGS.items()}
STEP = 1.9766                                    # Å, 50.6 nm over 256 px


def cal(mode: str) -> Path:
    """The calibration of a recording (tools/calibrate_parallax.py, via scripts/session_report.py)."""
    return CALIBRATION / f"{mode}.json"


def flags(mode: str) -> list[str]:
    return (CALIBRATION / f"{mode}.flags").read_text().split()


def geometry(mode: str) -> dict:
    """The parallax geometry run.py gets: the shift fit's, with C10 refined on the lattice."""
    c = json.loads(cal(mode).read_text())
    g = dict(c["geometry"])
    if (c.get("refine") or {}).get("used"):
        g["c10"] = c["refine"]["best_c10"]
    return g


def recon(mode: str, name: str) -> Path:
    """A run.py result folder: name = U<u>[_sub|_nodc|_fit|_live]."""
    return RECON / mode / name
