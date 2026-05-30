"""NVIDIA GPU VRAM pressure collector backed by NVML.

Reads ``nvmlDeviceGetMemoryInfo`` for every visible GPU.  This captures ALL
VRAM consumers on the device — model weights, KV cache, other processes,
CUDA runtime overhead — giving a true picture of available headroom rather
than just the engine's self-reported KV utilisation.

The same pynvml → ctypes fallback chain used by the thermal collector is
reproduced here so the two collectors remain independently usable without
requiring a shared NVML session.
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import importlib
from typing import Any, Protocol

from loguru import logger

from telemetry.memory.base import MemorySnapshot

NVML_SUCCESS = 0
NVML_TEMPERATURE_GPU = 0


class _NvmlClient(Protocol):
    name: str

    def device_count(self) -> int: ...
    def device_name(self, index: int) -> str | None: ...
    def memory_info(self, index: int) -> tuple[int, int] | None:
        """Return ``(used_bytes, total_bytes)`` or ``None`` on error."""
        ...

    def close(self) -> None: ...


class NvidiaVramCollector:
    """Polls NVIDIA NVML for per-GPU VRAM utilisation."""

    name = "nvidia-vram"

    def __init__(self, client: _NvmlClient, device_indices: list[int]) -> None:
        self._client = client
        self._device_indices = device_indices

    @classmethod
    def try_create(cls) -> NvidiaVramCollector | None:
        """Initialise NVML and return a collector, or ``None`` if unavailable."""
        for create_client in (_PynvmlClient.try_create, _CtypesNvmlClient.try_create):
            client = create_client()
            if client is None:
                continue
            try:
                count = client.device_count()
            except Exception as exc:
                logger.debug(
                    "NVIDIA NVML device count failed | backend={} error={}",
                    client.name,
                    exc,
                )
                client.close()
                continue
            if count <= 0:
                client.close()
                continue
            logger.info(
                "NVIDIA VRAM collector ready | backend={} deviceCount={}", client.name, count
            )
            return cls(client, list(range(count)))
        return None

    async def sample(self) -> MemorySnapshot | None:
        """Query all visible NVIDIA GPUs and return the most-pressured sample."""
        return await asyncio.to_thread(self._sample_sync)

    def _sample_sync(self) -> MemorySnapshot | None:
        worst: MemorySnapshot | None = None
        for index in self._device_indices:
            info = self._client.memory_info(index)
            if info is None:
                continue
            used, total = info
            pressure = used / total if total > 0 else 0.0
            name = self._client.device_name(index)
            logger.debug(
                "NVIDIA VRAM | index={} name={} usedBytes={} totalBytes={} pressure={:.3f}",
                index,
                name,
                used,
                total,
                pressure,
            )
            snapshot = MemorySnapshot(
                source=self.name,
                device_type="gpu",
                pressure=pressure,
                used_bytes=used,
                total_bytes=total,
                details={"index": index, "name": name},
            )
            if worst is None or snapshot.pressure > worst.pressure:
                worst = snapshot
        return worst

    async def close(self) -> None:
        await asyncio.to_thread(self._client.close)


# ── pynvml client ──────────────────────────────────────────────────────────


class _PynvmlClient:
    name = "nvidia-nvml"

    def __init__(self, pynvml: Any) -> None:
        self._pynvml = pynvml

    @classmethod
    def try_create(cls) -> _PynvmlClient | None:
        try:
            pynvml = importlib.import_module("pynvml")
        except ImportError:
            logger.debug("pynvml unavailable; trying direct NVML library")
            return None
        try:
            pynvml.nvmlInit()
        except Exception as exc:
            logger.debug("pynvml NVML init failed | error={}", exc)
            return None
        client = cls(pynvml)
        try:
            client.device_count()
        except Exception as exc:
            logger.debug("pynvml NVML device count failed after init | error={}", exc)
            client.close()
            return None
        return client

    def device_count(self) -> int:
        return int(self._pynvml.nvmlDeviceGetCount())

    def device_name(self, index: int) -> str | None:
        try:
            raw = self._pynvml.nvmlDeviceGetName(self._handle(index))
        except Exception:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    def memory_info(self, index: int) -> tuple[int, int] | None:
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle(index))
            return int(info.used), int(info.total)
        except Exception as exc:
            logger.debug("pynvml memory info failed | index={} error={}", index, exc)
            return None

    def close(self) -> None:
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass

    def _handle(self, index: int) -> Any:
        return self._pynvml.nvmlDeviceGetHandleByIndex(index)


# ── ctypes client ──────────────────────────────────────────────────────────


class _NvmlMemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _CtypesNvmlClient:
    name = "nvidia-nvml-ctypes"

    def __init__(self, nvml: Any, handles: list[ctypes.c_void_p]) -> None:
        self._nvml = nvml
        self._handles = handles

    @classmethod
    def try_create(cls) -> _CtypesNvmlClient | None:
        library_path = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        try:
            nvml = ctypes.CDLL(library_path)
        except OSError as exc:
            logger.debug(
                "Direct NVML library load failed | library={} error={}", library_path, exc
            )
            return None

        init_fn = _nvml_fn(nvml, "nvmlInit_v2") or _nvml_fn(nvml, "nvmlInit")
        if init_fn is None:
            return None
        init_fn.restype = ctypes.c_int
        if int(init_fn()) != NVML_SUCCESS:
            return None

        try:
            handles = _ctypes_device_handles(nvml)
        except Exception as exc:
            logger.debug("Direct NVML device discovery failed | error={}", exc)
            _ctypes_shutdown(nvml)
            return None
        if not handles:
            _ctypes_shutdown(nvml)
            return None
        return cls(nvml, handles)

    def device_count(self) -> int:
        return len(self._handles)

    def device_name(self, index: int) -> str | None:
        fn = _nvml_fn(self._nvml, "nvmlDeviceGetName")
        if fn is None:
            return None
        buf = ctypes.create_string_buffer(96)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        if int(fn(self._handles[index], buf, len(buf))) != NVML_SUCCESS:
            return None
        return buf.value.decode("utf-8", errors="replace")

    def memory_info(self, index: int) -> tuple[int, int] | None:
        fn = _nvml_fn(self._nvml, "nvmlDeviceGetMemoryInfo")
        if fn is None:
            return None
        info = _NvmlMemoryInfo()
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemoryInfo)]
        fn.restype = ctypes.c_int
        if int(fn(self._handles[index], ctypes.byref(info))) != NVML_SUCCESS:
            return None
        return int(info.used), int(info.total)

    def close(self) -> None:
        _ctypes_shutdown(self._nvml)


# ── helpers ────────────────────────────────────────────────────────────────


def _ctypes_device_handles(nvml: Any) -> list[ctypes.c_void_p]:
    count_fn = _nvml_fn(nvml, "nvmlDeviceGetCount_v2") or _nvml_fn(nvml, "nvmlDeviceGetCount")
    handle_fn = _nvml_fn(nvml, "nvmlDeviceGetHandleByIndex_v2") or _nvml_fn(
        nvml, "nvmlDeviceGetHandleByIndex"
    )
    if count_fn is None or handle_fn is None:
        return []
    count = ctypes.c_uint()
    count_fn.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    count_fn.restype = ctypes.c_int
    if int(count_fn(ctypes.byref(count))) != NVML_SUCCESS:
        return []
    handles: list[ctypes.c_void_p] = []
    handle_fn.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    handle_fn.restype = ctypes.c_int
    for i in range(count.value):
        h = ctypes.c_void_p()
        if int(handle_fn(i, ctypes.byref(h))) == NVML_SUCCESS:
            handles.append(h)
    return handles


def _ctypes_shutdown(nvml: Any) -> None:
    fn = _nvml_fn(nvml, "nvmlShutdown")
    if fn:
        fn.restype = ctypes.c_int
        fn()


def _nvml_fn(nvml: Any, name: str) -> Any | None:
    try:
        return getattr(nvml, name)
    except AttributeError:
        return None
