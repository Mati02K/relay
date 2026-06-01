#!/usr/bin/env python3
"""Scapegoat memory load for the `memory` scenario — run on the Mac (MLX) worker.

Holds a chunk of **incompressible, resident** RAM so macOS unified-memory
pressure rises, which the worker's ``telemetry/memory/apple.py`` collector reports
as ``mw``. With the memory scenario's signal on (``memory=1``), the scheduler then
routes away from this node — that's the difference the test measures.

Usage
-----
    python3 mac_memory_load.py [GB]        # default 8 GB

Size GB to ~60–80% of the machine's *free* RAM so pressure climbs without driving
it into heavy swap. Random bytes are used so macOS memory compression can't
silently reclaim the pages. Ctrl-C to release.
"""

from __future__ import annotations

import os
import sys
import time

DEFAULT_GB = 8.0
CHUNK_BYTES = 256 * 1024 * 1024  # allocate in 256 MB pieces for progress + safety


def main() -> None:
    gb = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GB
    target = int(gb * (1024**3))
    print(f"Allocating ~{gb:.1f} GB of incompressible RAM (random bytes)…")

    chunks: list[bytearray] = []
    held = 0
    try:
        while held < target:
            take = min(CHUNK_BYTES, target - held)
            chunks.append(bytearray(os.urandom(take)))  # random → not compressible
            held += take
            print(f"  held {held // (1024 ** 2)} MB", end="\r", flush=True)
    except MemoryError:
        print(f"\nMemoryError at {held // (1024 ** 2)} MB — that's plenty of pressure.")

    print(f"\nHolding {held // (1024 ** 2)} MB resident. Watch `mw` on the coordinator's "
          "/v1/workers; Ctrl-C to release.")
    try:
        while True:
            for chunk in chunks:  # touch a byte per chunk so pages stay hot/resident
                chunk[0] ^= 1
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nReleased.")


if __name__ == "__main__":
    main()
