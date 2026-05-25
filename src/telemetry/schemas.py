"""Pydantic telemetry schemas shared by workers, engines, and schedulers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from telemetry.prefix_cache import (
    DEFAULT_PREFIX_BLOCK_SIZE_TOKENS,
    DEFAULT_PREFIX_BYTES_PER_TOKEN,
    DEFAULT_PREFIX_CACHE_BLOCKS,
    DEFAULT_PREFIX_HASH_SCHEME,
    DEFAULT_PREFIX_TOKENIZER_ID,
    PrefixHashConfig,
)


class EngineReportedTelemetry(BaseModel):
    """Telemetry read directly from the inference backend.

    For llama.cpp server this comes from endpoints such as ``/metrics``.
    """

    model_config = {"frozen": True}

    qw: int = Field(
        0,
        description="Current engine queue/load; engine-provided when available, otherwise 0",
    )
    mw: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Memory or KV-cache pressure, where 0.0 is empty/low and 1.0 is full/high",
    )


class PrefixCacheTelemetry(BaseModel):
    """Prefix-cache hashes that a worker believes are useful for routing.

    These hashes are scheduler hints. They do not expose prompt content, and
    they must only be compared when model, tokenizer, hash scheme, and block
    size match.
    """

    model_config = {"frozen": True}

    hash_scheme: str = Field(
        DEFAULT_PREFIX_HASH_SCHEME,
        description="Algorithm used to compute prefix block hashes",
    )
    model_id: str = Field("unknown", description="Model identity used in the hash root")
    tokenizer_id: str = Field(
        DEFAULT_PREFIX_TOKENIZER_ID,
        description="Tokenizer identity used in the hash root",
    )
    block_size_tokens: int = Field(
        DEFAULT_PREFIX_BLOCK_SIZE_TOKENS,
        gt=0,
        description="Approximate or exact token count per prefix-cache block",
    )
    bytes_per_token: float = Field(
        DEFAULT_PREFIX_BYTES_PER_TOKEN,
        gt=0.0,
        description=(
            "Approximation used by text fallback hashing; exact token hashing can ignore it"
        ),
    )
    max_blocks: int = Field(
        DEFAULT_PREFIX_CACHE_BLOCKS,
        gt=0,
        description="Maximum recent prefix blocks retained by the worker",
    )
    block_hashes: list[str] = Field(
        default_factory=list,
        description="Recent chained prefix block hashes published by this worker",
    )

    @classmethod
    def from_config(
        cls,
        config: PrefixHashConfig,
        block_hashes: list[str],
    ) -> PrefixCacheTelemetry:
        """Build telemetry from a prefix hash config and observed block hashes."""
        return cls(
            hash_scheme=config.hash_scheme,
            model_id=config.model_id,
            tokenizer_id=config.tokenizer_id,
            block_size_tokens=config.block_size_tokens,
            bytes_per_token=config.bytes_per_token,
            max_blocks=config.max_blocks,
            block_hashes=list(block_hashes),
        )

    def to_hash_config(self) -> PrefixHashConfig:
        """Return the hash config needed to compute comparable request hashes."""
        return PrefixHashConfig(
            hash_scheme=self.hash_scheme,
            model_id=self.model_id,
            tokenizer_id=self.tokenizer_id,
            block_size_tokens=self.block_size_tokens,
            bytes_per_token=self.bytes_per_token,
            max_blocks=self.max_blocks,
        )


class RequestComputedTelemetry(BaseModel):
    """Telemetry calculated by Relay around each generation call."""

    model_config = {"frozen": True}

    sw_by_bucket: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Recent decode speed in output tokens/sec, grouped by prompt-length bucket "
            "such as '<=256' or '<=1024'"
        ),
    )
    prefix_cache: PrefixCacheTelemetry = Field(
        default_factory=lambda: PrefixCacheTelemetry.from_config(PrefixHashConfig.from_env(), []),
        description="Recent prefix-cache block hashes published by this worker",
    )
    sprefill_tokens_per_sec: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Recent prompt prefill speed in input tokens/sec, estimated from time to first token"
        ),
    )


class SystemTelemetry(BaseModel):
    """Telemetry supplied by worker-level system collectors outside the inference backend."""

    model_config = {"frozen": True}

    jw: float = Field(
        0.0,
        ge=0.0,
        description="Network round-trip jitter in milliseconds, usually filled from heartbeats",
    )
    theta_w: int = Field(
        0,
        ge=0,
        le=1,
        description="Thermal throttling flag: 0 means normal, 1 means throttled",
    )


class Telemetry(BaseModel):
    """Combined runtime measurements reported by an inference worker.

    The final coordinator-facing snapshot is built from backend metrics,
    Python request-computed metrics, and worker-level system metrics.
    """

    model_config = {"frozen": False}

    qw: int = Field(
        0,
        description="Current engine queue/load; engine-provided when available, otherwise 0",
    )
    sw_by_bucket: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Recent decode speed in output tokens/sec, grouped by prompt-length bucket "
            "such as '<=256' or '<=1024'"
        ),
    )
    mw: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Memory or KV-cache pressure, where 0.0 is empty/low and 1.0 is full/high",
    )
    jw: float = Field(
        0.0,
        ge=0.0,
        description="Network round-trip jitter in milliseconds, usually filled from heartbeats",
    )
    theta_w: int = Field(
        0,
        ge=0,
        le=1,
        description="Thermal throttling flag: 0 means normal, 1 means throttled",
    )
    prefix_cache: PrefixCacheTelemetry = Field(
        default_factory=lambda: PrefixCacheTelemetry.from_config(PrefixHashConfig.from_env(), []),
        description="Recent prefix-cache block hashes published by this worker",
    )
    sprefill_tokens_per_sec: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Recent prompt prefill speed in input tokens/sec, estimated from time to first token"
        ),
    )

    @classmethod
    def from_parts(
        cls,
        *,
        engine: EngineReportedTelemetry | None = None,
        request: RequestComputedTelemetry | None = None,
        system: SystemTelemetry | None = None,
    ) -> Telemetry:
        """Merge telemetry from backend, request, and system sources."""
        engine = engine or EngineReportedTelemetry(qw=0, mw=0.0)
        request = request or RequestComputedTelemetry(
            sw_by_bucket={},
            prefix_cache=PrefixCacheTelemetry.from_config(PrefixHashConfig.from_env(), []),
            sprefill_tokens_per_sec=0.0,
        )
        system = system or SystemTelemetry(jw=0.0, theta_w=0)
        return cls(
            qw=engine.qw,
            sw_by_bucket=dict(request.sw_by_bucket),
            mw=engine.mw,
            jw=system.jw,
            theta_w=system.theta_w,
            prefix_cache=request.prefix_cache,
            sprefill_tokens_per_sec=request.sprefill_tokens_per_sec,
        )
