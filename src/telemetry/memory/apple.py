"""macOS unified-memory pressure collector.

Apple Silicon shares a single physical memory pool between the CPU and GPU.
There is no separate VRAM budget, so the right signal is system-wide memory
pressure rather than a device-specific VRAM ratio.

Two ``sysctl`` reads provide total and page-size; ``vm_stat`` provides the
page counts.  All three are non-privileged and available on any macOS
version supported by Relay.

Pressure is:

    used_bytes = (active + wired + compressed + occupied) * page_size
    pressure   = used_bytes / hw.memsize

``inactive`` pages are excluded from "used" because the kernel reclaims them
on demand — they are semantically available even though they are occupied.
"""

from __future__ import annotations

import asyncio
import re

from loguru import logger

from telemetry.memory.base import MemorySnapshot

_PAGESIZE_CMD = ("sysctl", "-n", "hw.pagesize")
_MEMSIZE_CMD = ("sysctl", "-n", "hw.memsize")
_VMSTAT_CMD = ("vm_stat",)
_PROBE_TIMEOUT_SECONDS = 3.0

_VM_STAT_RE = re.compile(r"^Pages\s+(\w[\w ]*\w|\w+):\s+([\d.]+)\.$", re.MULTILINE)


class MacOSMemoryCollector:
    """Reads macOS unified memory pressure via ``sysctl`` + ``vm_stat``."""

    name = "macos-memory"

    @classmethod
    def try_create(cls) -> MacOSMemoryCollector | None:
        """Return a collector on macOS; ``None`` on other platforms."""
        import shutil

        if not shutil.which("vm_stat") or not shutil.which("sysctl"):
            logger.debug("vm_stat or sysctl missing — macOS memory collector disabled")
            return None
        return cls()

    async def sample(self) -> MemorySnapshot | None:
        """Return unified memory pressure or ``None`` on subprocess failure."""
        try:
            page_size, memsize, vm_output = await asyncio.wait_for(
                asyncio.gather(
                    _run_command(_PAGESIZE_CMD),
                    _run_command(_MEMSIZE_CMD),
                    _run_command(_VMSTAT_CMD),
                ),
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.debug("macOS memory: sysctl/vm_stat timed out")
            return None
        except Exception as exc:
            logger.debug("macOS memory: subprocess error | error={}", exc)
            return None

        if page_size is None or memsize is None or vm_output is None:
            return None

        try:
            page_size_bytes = int(page_size.strip())
            total_bytes = int(memsize.strip())
        except ValueError as exc:
            logger.debug("macOS memory: sysctl parse error | error={}", exc)
            return None

        if total_bytes <= 0 or page_size_bytes <= 0:
            return None

        pages = _parse_vm_stat(vm_output)
        used_pages = (
            pages.get("active", 0)
            + pages.get("wired down", 0)
            + pages.get("occupied by compressor", 0)
        )
        used_bytes = used_pages * page_size_bytes
        pressure = min(1.0, used_bytes / total_bytes)

        logger.debug(
            "macOS memory sample | usedBytes={} totalBytes={} pressure={:.3f} pages={}",
            used_bytes,
            total_bytes,
            pressure,
            pages,
        )
        return MemorySnapshot(
            source=self.name,
            device_type="unified",
            pressure=pressure,
            used_bytes=used_bytes,
            total_bytes=total_bytes,
            details={"page_counts": pages},
        )

    async def close(self) -> None:
        """No persistent resources."""
        return None


def _parse_vm_stat(output: str) -> dict[str, int]:
    """Extract page counts from ``vm_stat`` output."""
    result: dict[str, int] = {}
    for match in _VM_STAT_RE.finditer(output):
        key = match.group(1).strip().lower()
        try:
            result[key] = int(match.group(2))
        except ValueError:
            continue
    return result


async def _run_command(cmd: tuple[str, ...]) -> str | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        return stdout.decode(errors="replace")
    except (OSError, FileNotFoundError):
        return None
