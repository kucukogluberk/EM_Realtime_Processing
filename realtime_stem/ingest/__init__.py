"""Ingest stage: receive detector frames and hand decoded batches to an `Accumulator`.

Frames arrive over ZeroMQ from the DECTRIS detector control unit and are bslz4-decompressed on
the GPU by two compiled Holoscan C++ operators — `ZmqRxOp` and `DecompressGpuOp` — from
`operators/`, compiled into the image by `docker/Dockerfile` (see the README). The Python
`GenericSinkOp` receives each decoded batch as a zero-copy CUDA tensor (DLPack) and calls
`accumulator.process(frames, ids)`. Nothing here is parallax-specific.
"""
