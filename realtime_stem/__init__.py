"""Real-time STEM processing: a modular ingest → processing → stream-out pipeline.

`ingest` receives and GPU-decompresses detector frames (Holoscan C++ ops), `processing`
turns batches of frames into a reconstruction behind a small `Accumulator` contract, and
`streaming` publishes the reconstruction-so-far over ZeroMQ. Parallax / tcBF is the worked
example; new algorithms drop in behind the same contract (see `processing.registry`).
"""

__version__ = "2.0.2"
