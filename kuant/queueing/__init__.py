"""kuant.queueing — hardware detection + chunk sizing + (future) job queue.

Public exports:
  DEVICE     — DeviceProfile snapshot detected at import time
  THROTTLE   — ChunkSizer singleton used by kernels for chunk decisions
  detect_hw  — re-run device detection (rarely needed)

Design note: keeping DEVICE and THROTTLE as module-level singletons is
intentional. It matches the process model — one process, one device profile,
one throttle. Cross-process coordination (if ever needed) belongs in a
separate layer, not here.
"""

from .hardware import DEVICE, DeviceProfile, detect_hw
from .throttle import THROTTLE, ChunkSizer

__all__ = ["DEVICE", "DeviceProfile", "detect_hw", "THROTTLE", "ChunkSizer"]
