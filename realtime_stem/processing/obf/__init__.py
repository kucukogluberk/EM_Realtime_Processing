"""Optimum bright field (OBF) direct ptychography, a live channel next to parallax.

`core` is the method: the correlation route of quantem's `DirectPtychographyMontage`, ported to
torch on the parallax geometry (`ObfSetup`: the probe, the canvas layout and the Gram;
`reconstruct_once`: a whole `latest` store in one call).
"""

from realtime_stem.processing.obf.core import ObfSetup, reconstruct_once

__all__ = ["ObfSetup", "reconstruct_once"]
