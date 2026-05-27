"""NVIDIA GPU thermal-pressure collector backed by NVML.

NVML is the NVIDIA driver API used for GPU telemetry. Relay prefers the
``pynvml`` module from the ``nvidia-ml-py`` package, and falls back to loading
``libnvidia-ml.so.1`` directly with ``ctypes`` when that package is missing.

The collector reads GPU temperature, thermal limits, and NVML clock-event
reason bits. Only thermal slowdown bits are treated as throttling; idle
downclocking and power caps are not heat-driven and are ignored here.
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import importlib
from typing import Any, Protocol

from loguru import logger

from telemetry.thermal.base import (
    ThermalSourceSample,
    pressure_from_temperature,
    pressure_to_state,
)

NVML_SUCCESS = 0
NVML_TEMPERATURE_GPU = 0
NVML_TEMPERATURE_THRESHOLD_SHUTDOWN = 0
NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1

# Bit values from nvml.h. Keeping them local lets this module import even when
# no NVIDIA driver or Python NVML package exists.
#
# Note: NVML also exposes a generic HW_SLOWDOWN bit (0x8) which is set when
# any hardware slowdown is engaged — including power brake and sync boost,
# both of which fire for non-thermal reasons on consumer/laptop GPUs. We
# intentionally exclude it from the thermal mask to avoid false-positive
# throttling alerts on idle systems. Only the two specifically-thermal
# bits below qualify.
_REASON_SW_THERMAL_SLOWDOWN = 0x0000000000000020
_REASON_HW_THERMAL_SLOWDOWN = 0x0000000000000040

_THERMAL_REASON_MASK = _REASON_SW_THERMAL_SLOWDOWN | _REASON_HW_THERMAL_SLOWDOWN

# Laptop drivers assert SW_THERMAL_SLOWDOWN as part of routine power management
# even when the GPU is idle and cool, so the bit alone is unreliable. Require
# the temperature to be within this margin of the trip limit before treating
# a SW slowdown as a real thermal event.
_SW_THERMAL_CORROBORATION_MARGIN_C = 10.0


def is_thermal_throttle(reason_bits: int) -> int:
    """Return 1 if the NVML reason bitmask carries any thermal slowdown flag."""
    return 1 if (reason_bits & _THERMAL_REASON_MASK) else 0


def is_hw_thermal_throttle(reason_bits: int) -> bool:
    """Return True if NVML's HW thermal slowdown bit is set."""
    return bool(reason_bits & _REASON_HW_THERMAL_SLOWDOWN)


def is_sw_thermal_throttle(reason_bits: int) -> bool:
    """Return True if NVML's SW thermal slowdown bit is set."""
    return bool(reason_bits & _REASON_SW_THERMAL_SLOWDOWN)


def is_thermal_throttle_corroborated(
    reason_bits: int,
    temperature_c: float | None,
    limit_c: float | None,
) -> bool:
    """Decide whether a slowdown bit reflects genuine thermal throttling.

    HW thermal slowdown is always trusted. SW thermal slowdown is only trusted
    when ``temperature_c`` is within ``_SW_THERMAL_CORROBORATION_MARGIN_C`` of
    ``limit_c`` — laptop drivers raise the SW bit for routine power management
    even at idle temperatures.
    """
    if is_hw_thermal_throttle(reason_bits):
        return True
    if not is_sw_thermal_throttle(reason_bits):
        return False
    if temperature_c is None or limit_c is None:
        return False
    return temperature_c >= limit_c - _SW_THERMAL_CORROBORATION_MARGIN_C


class _NvmlClient(Protocol):
    """Small backend interface shared by pynvml and direct-ctypes NVML."""

    name: str

    def device_count(self) -> int:
        """Return visible GPU count."""
        ...

    def device_name(self, index: int) -> str | None:
        """Return the GPU name, when available."""
        ...

    def temperature_c(self, index: int) -> float | None:
        """Return GPU temperature in Celsius."""
        ...

    def thermal_limit_c(self, index: int) -> float | None:
        """Return the thermal slowdown/shutdown threshold in Celsius."""
        ...

    def throttle_reason_bits(self, index: int) -> int | None:
        """Return NVML current clock-event reason bits."""
        ...

    def close(self) -> None:
        """Release NVML resources."""
        ...


class NvidiaNvmlCollector:
    """Polls NVIDIA NVML for GPU temperature and thermal-slowdown state."""

    def __init__(self, client: _NvmlClient, device_indices: list[int]) -> None:
        self.name = client.name
        self._client = client
        self._device_indices = device_indices

    @classmethod
    def try_create(cls) -> NvidiaNvmlCollector | None:
        """Initialize NVML and return a collector, or ``None`` if unavailable."""
        for create_client in (_PynvmlClient.try_create, _CtypesNvmlClient.try_create):
            client = create_client()
            if client is None:
                continue
            try:
                count = client.device_count()
            except Exception as e:
                logger.debug(
                    "NVIDIA NVML device count failed after init | backend={} error={}",
                    client.name,
                    e,
                )
                client.close()
                continue
            if count <= 0:
                client.close()
                continue
            logger.info(
                "NVIDIA NVML thermal collector ready | backend={} deviceCount={}",
                client.name,
                count,
            )
            return cls(client, list(range(count)))
        return None

    async def sample(self) -> list[ThermalSourceSample]:
        """Query every visible NVIDIA GPU and return normalized pressure samples."""
        return await asyncio.to_thread(self._sample_sync)

    def _sample_sync(self) -> list[ThermalSourceSample]:
        samples: list[ThermalSourceSample] = []
        for index in self._device_indices:
            name = self._client.device_name(index)
            reasons = self._client.throttle_reason_bits(index)
            temp_c = self._client.temperature_c(index)
            limit_c = self._client.thermal_limit_c(index)
            reason_bits = reasons or 0
            hw_throttle = is_hw_thermal_throttle(reason_bits)
            sw_throttle = is_sw_thermal_throttle(reason_bits)
            thermal_throttle = is_thermal_throttle_corroborated(reason_bits, temp_c, limit_c)
            temp_pressure = pressure_from_temperature(temp_c, limit_c)
            pressure = max(temp_pressure, 0.85 if thermal_throttle else 0.0)
            logger.debug(
                "NVIDIA NVML | backend={} index={} name={} tempC={} limitC={} "
                "reasonBits={} hwThrottle={} swThrottle={} thermalThrottle={} pressure={:.3f}",
                self.name,
                index,
                name,
                temp_c,
                limit_c,
                reason_bits,
                hw_throttle,
                sw_throttle,
                thermal_throttle,
                pressure,
            )
            samples.append(
                ThermalSourceSample(
                    source=self.name,
                    device_type="gpu",
                    pressure=pressure,
                    state=pressure_to_state(pressure),
                    temperature_c=temp_c,
                    limit_c=limit_c,
                    throttle_active=thermal_throttle,
                    details={
                        "index": index,
                        "name": name,
                        "reason_bits": reasons,
                    },
                )
            )
        return samples

    async def close(self) -> None:
        """Shut down NVML."""
        await asyncio.to_thread(self._client.close)

    def close_sync(self) -> None:
        """Shut down NVML from synchronous CLI probes."""
        self._client.close()


class _PynvmlClient:
    """NVML client using the ``pynvml`` module from ``nvidia-ml-py``."""

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
        except Exception as e:
            logger.debug("pynvml NVML init failed | error={}", e)
            return None

        client = cls(pynvml)
        try:
            client.device_count()
        except Exception as e:
            logger.debug("pynvml NVML device count failed | error={}", e)
            client.close()
            return None
        return client

    def device_count(self) -> int:
        return int(self._pynvml.nvmlDeviceGetCount())

    def device_name(self, index: int) -> str | None:
        try:
            raw_name = self._pynvml.nvmlDeviceGetName(self._handle(index))
        except Exception as e:
            logger.debug("pynvml name read failed | index={} error={}", index, e)
            return None
        if isinstance(raw_name, bytes):
            return raw_name.decode("utf-8", errors="replace")
        return str(raw_name)

    def temperature_c(self, index: int) -> float | None:
        try:
            return float(
                self._pynvml.nvmlDeviceGetTemperature(
                    self._handle(index),
                    getattr(self._pynvml, "NVML_TEMPERATURE_GPU", NVML_TEMPERATURE_GPU),
                )
            )
        except Exception as e:
            logger.debug("pynvml temperature read failed | index={} error={}", index, e)
            return None

    def thermal_limit_c(self, index: int) -> float | None:
        getter = getattr(self._pynvml, "nvmlDeviceGetTemperatureThreshold", None)
        if getter is None:
            return None
        threshold_names = (
            "NVML_TEMPERATURE_THRESHOLD_SLOWDOWN",
            "NVML_TEMPERATURE_THRESHOLD_SHUTDOWN",
        )
        for name in threshold_names:
            threshold = getattr(self._pynvml, name, None)
            if threshold is None:
                continue
            try:
                return float(getter(self._handle(index), threshold))
            except Exception:
                continue
        return None

    def throttle_reason_bits(self, index: int) -> int | None:
        getter = getattr(
            self._pynvml,
            "nvmlDeviceGetCurrentClocksEventReasons",
            None,
        ) or getattr(
            self._pynvml,
            "nvmlDeviceGetCurrentClocksThrottleReasons",
            None,
        )
        if getter is None:
            return None
        try:
            return int(getter(self._handle(index)))
        except Exception as e:
            logger.debug("pynvml throttle reason read failed | index={} error={}", index, e)
            return None

    def close(self) -> None:
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass

    def _handle(self, index: int) -> Any:
        return self._pynvml.nvmlDeviceGetHandleByIndex(index)


class _CtypesNvmlClient:
    """NVML client using ``ctypes`` against the local NVIDIA driver library."""

    name = "nvidia-nvml-ctypes"

    def __init__(self, nvml: Any, handles: list[ctypes.c_void_p]) -> None:
        self._nvml = nvml
        self._handles = handles

    @classmethod
    def try_create(cls) -> _CtypesNvmlClient | None:
        library_path = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        try:
            nvml = ctypes.CDLL(library_path)
        except OSError as e:
            logger.debug("direct NVML library load failed | library={} error={}", library_path, e)
            return None

        init = _get_nvml_function(nvml, "nvmlInit_v2") or _get_nvml_function(nvml, "nvmlInit")
        if init is None:
            logger.debug("direct NVML init function missing | library={}", library_path)
            return None
        init.restype = ctypes.c_int
        rc = int(init())
        if rc != NVML_SUCCESS:
            logger.debug(
                "direct NVML init failed | rc={} error={}",
                rc,
                _nvml_error_string(nvml, rc),
            )
            return None

        try:
            handles = _ctypes_device_handles(nvml)
        except Exception as e:
            logger.debug("direct NVML device discovery failed | error={}", e)
            _ctypes_shutdown(nvml)
            return None
        if not handles:
            _ctypes_shutdown(nvml)
            return None
        return cls(nvml, handles)

    def device_count(self) -> int:
        return len(self._handles)

    def device_name(self, index: int) -> str | None:
        fn = _get_nvml_function(self._nvml, "nvmlDeviceGetName")
        if fn is None:
            return None
        buffer = ctypes.create_string_buffer(96)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        rc = int(fn(self._handles[index], buffer, len(buffer)))
        if rc != NVML_SUCCESS:
            logger.debug(
                "direct NVML name read failed | index={} rc={} error={}",
                index,
                rc,
                _nvml_error_string(self._nvml, rc),
            )
            return None
        return buffer.value.decode("utf-8", errors="replace")

    def temperature_c(self, index: int) -> float | None:
        fn = _get_nvml_function(self._nvml, "nvmlDeviceGetTemperature")
        if fn is None:
            return None
        value = ctypes.c_uint()
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        fn.restype = ctypes.c_int
        rc = int(fn(self._handles[index], NVML_TEMPERATURE_GPU, ctypes.byref(value)))
        if rc != NVML_SUCCESS:
            logger.debug(
                "direct NVML temperature read failed | index={} rc={} error={}",
                index,
                rc,
                _nvml_error_string(self._nvml, rc),
            )
            return None
        return float(value.value)

    def thermal_limit_c(self, index: int) -> float | None:
        fn = _get_nvml_function(self._nvml, "nvmlDeviceGetTemperatureThreshold")
        if fn is None:
            return None
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        fn.restype = ctypes.c_int
        for threshold in (
            NVML_TEMPERATURE_THRESHOLD_SLOWDOWN,
            NVML_TEMPERATURE_THRESHOLD_SHUTDOWN,
        ):
            value = ctypes.c_uint()
            rc = int(fn(self._handles[index], threshold, ctypes.byref(value)))
            if rc == NVML_SUCCESS:
                return float(value.value)
        return None

    def throttle_reason_bits(self, index: int) -> int | None:
        fn = _get_nvml_function(
            self._nvml,
            "nvmlDeviceGetCurrentClocksEventReasons",
        ) or _get_nvml_function(
            self._nvml,
            "nvmlDeviceGetCurrentClocksThrottleReasons",
        )
        if fn is None:
            return None
        value = ctypes.c_ulonglong()
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
        fn.restype = ctypes.c_int
        rc = int(fn(self._handles[index], ctypes.byref(value)))
        if rc != NVML_SUCCESS:
            logger.debug(
                "direct NVML throttle reason read failed | index={} rc={} error={}",
                index,
                rc,
                _nvml_error_string(self._nvml, rc),
            )
            return None
        return int(value.value)

    def close(self) -> None:
        _ctypes_shutdown(self._nvml)


def _ctypes_device_handles(nvml: Any) -> list[ctypes.c_void_p]:
    count_fn = _get_nvml_function(nvml, "nvmlDeviceGetCount_v2") or _get_nvml_function(
        nvml,
        "nvmlDeviceGetCount",
    )
    handle_fn = _get_nvml_function(
        nvml,
        "nvmlDeviceGetHandleByIndex_v2",
    ) or _get_nvml_function(
        nvml,
        "nvmlDeviceGetHandleByIndex",
    )
    if count_fn is None or handle_fn is None:
        return []

    count = ctypes.c_uint()
    count_fn.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    count_fn.restype = ctypes.c_int
    rc = int(count_fn(ctypes.byref(count)))
    if rc != NVML_SUCCESS:
        logger.debug(
            "direct NVML device count failed | rc={} error={}",
            rc,
            _nvml_error_string(nvml, rc),
        )
        return []

    handles: list[ctypes.c_void_p] = []
    handle_fn.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    handle_fn.restype = ctypes.c_int
    for index in range(count.value):
        handle = ctypes.c_void_p()
        rc = int(handle_fn(index, ctypes.byref(handle)))
        if rc != NVML_SUCCESS:
            logger.debug(
                "direct NVML handle read failed | index={} rc={} error={}",
                index,
                rc,
                _nvml_error_string(nvml, rc),
            )
            continue
        handles.append(handle)
    return handles


def _ctypes_shutdown(nvml: Any) -> None:
    shutdown = _get_nvml_function(nvml, "nvmlShutdown")
    if shutdown is None:
        return
    shutdown.restype = ctypes.c_int
    shutdown()


def _get_nvml_function(nvml: Any, name: str) -> Any | None:
    try:
        return getattr(nvml, name)
    except AttributeError:
        return None


def _nvml_error_string(nvml: Any, rc: int) -> str:
    error_string = _get_nvml_function(nvml, "nvmlErrorString")
    if error_string is None:
        return "unknown"
    try:
        error_string.argtypes = [ctypes.c_int]
        error_string.restype = ctypes.c_char_p
        raw = error_string(rc)
    except Exception:
        return "unknown"
    if not raw:
        return "unknown"
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)
