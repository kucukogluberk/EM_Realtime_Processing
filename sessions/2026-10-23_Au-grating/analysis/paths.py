"""Where everything of the 2026-10-23 Au-grating session lives (sessions/README.md has the layout)."""
import json
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
SESSION = ANALYSIS.parent                        # sessions/2026-10-23_Au-grating
SESSIONS = SESSION.parent
REPO = SESSIONS.parent
CALIBRATION = SESSION / "calibration"            # <id>.flags / .json (scripts/session_report.py)
RECON = SESSION / "recon"                        # <id>/U<u>[_fit|_live] (not in git)
DATA = SESSION / "data"
FIGURES = SESSION / "figures"                    # the downscaled figures kept in git
FULL = FIGURES / "full"                          # full-resolution figures (not in git)
OCT20 = SESSIONS / "2026-10-20_Gd2O3"            # the session these are compared with

_session = json.loads((SESSION / "session.json").read_text())
RECORDINGS = {m: SESSION / r["file"] for m, r in _session["recordings"].items()}
SNAPSHOTS = sorted((DATA / "Initial_test_A1").glob("snapshot*_phase_flipped.npy"))
AU = {"111": 2.355, "200": 2.039, "220": 1.442}  # Å, a = 4.078 Å


def cal(rid: str, session: Path = SESSION) -> dict:
    return json.loads((session / "calibration" / f"{rid}.json").read_text())


def recon(rid: str, name: str, session: Path = SESSION) -> Path:
    return session / "recon" / rid / name


def oct20_recording(rid: str) -> Path:
    s = json.loads((OCT20 / "session.json").read_text())
    return OCT20 / s["recordings"][rid]["file"]
