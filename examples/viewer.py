"""Live viewer: subscribe to a `run.py --live-endpoint` PUB and redraw as the scan fills in.

The reconstruction "wipes in" during the scan because the sink publishes a throttled snapshot of
the reconstruction-so-far (~5 Hz). This viewer keeps only the newest frame (CONFLATE), so it never
lags behind the pipeline.

    python examples/viewer.py tcp://127.0.0.1:5601
    python examples/viewer.py tcp://127.0.0.1:5601 --high-pass 25   # subtract a slow background

One example consumer of the stream — the pipeline doesn't depend on it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root on path

from examples._display import high_pass, robust_clim   # noqa: E402


def _log(msg: str) -> None:
    print(f"[viewer {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _View:
    """Holds the matplotlib state and renders frames."""

    def __init__(self, label: str, hp_sigma: float):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.label = label
        self.hp = hp_sigma
        self.hp_note = f"  |  high-pass σ={hp_sigma:g}px" if hp_sigma > 0 else ""
        self.fig, self.ax = plt.subplots(figsize=(7.5, 7.5))
        try:
            self.fig.canvas.manager.set_window_title(f"live view — {label}")
        except Exception:
            pass
        self.im = None
        self.cbar = None
        self.n = 0
        plt.ion()
        plt.show(block=False)
        self.ax.set_title(f"waiting for {label} …")
        self.fig.canvas.draw()

    def alive(self) -> bool:
        return self.plt.fignum_exists(self.fig.number)

    def pump(self, seconds: float) -> None:
        self.fig.canvas.flush_events()
        self.fig.canvas.start_event_loop(seconds)

    def render(self, raw) -> None:
        disp = high_pass(raw, self.hp) if self.hp > 0 else raw
        lo, hi = robust_clim(disp)
        self.n += 1
        if self.im is None or disp.shape != self.im.get_array().shape:
            self.ax.clear()
            self.im = self.ax.imshow(disp, cmap="gray", vmin=lo, vmax=hi)
            if self.cbar is None:
                self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046)
            else:
                self.cbar.update_normal(self.im)
        else:
            self.im.set_data(disp)
            self.im.set_clim(lo, hi)
        self.ax.set_title(f"{self.label} — update #{self.n} "
                          f"at {time.strftime('%H:%M:%S')}{self.hp_note}")
        self.fig.canvas.draw()
        _log(f"update #{self.n}: shape={disp.shape} range=[{lo:.4g}, {hi:.4g}]")


def run(endpoint: str, label: str, hp_sigma: float) -> None:
    from realtime_stem.streaming.live_stream import LiveSubscriber   # lazy: needs pyzmq
    view = _View(label, hp_sigma)
    sub = LiveSubscriber(endpoint)
    _log(f"subscribed to {endpoint}")
    while view.alive():
        try:
            msg = sub.recv_latest(timeout_ms=100)
            if msg is not None:
                view.render(msg[0])
            view.pump(0.01)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            _log(f"cycle error ({type(exc).__name__}: {exc}); continuing")
            time.sleep(0.2)
    sub.close()
    _log("window closed — exiting")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("endpoint", help="tcp://host:port of the run.py --live-endpoint PUB.")
    p.add_argument("--high-pass", type=float, default=0.0, metavar="SIGMA_PX",
                   help="Gaussian high-pass sigma in pixels (0 = off).")
    p.add_argument("--label", default=None, help="Window/title label.")
    args = p.parse_args()
    run(args.endpoint, args.label or args.endpoint, args.high_pass)


if __name__ == "__main__":
    main()
