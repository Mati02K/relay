"""Persistent Relay CLI configuration."""

from __future__ import annotations

import json
import socket
import uuid
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from relay.paths import RelayPaths

NodeRole = Literal["coordinator", "worker", "dual"]
NetworkBackend = Literal["lan", "tailscale"]
MembershipBackend = Literal["etcd"]
EngineName = Literal["llama.cpp"]


class ConfigError(RuntimeError):
    """Raised when Relay CLI config is missing or invalid."""


class NetworkConfig(BaseModel):
    """Network backend selected for cluster traffic."""

    backend: NetworkBackend = "lan"


class MembershipConfig(BaseModel):
    """Membership backend and local process settings."""

    backend: MembershipBackend = "etcd"
    host: str = "127.0.0.1"
    grpc_port: int = 50051
    service_bin: str | None = None
    etcd_bin: str | None = None
    etcd_endpoints: str = "127.0.0.1:2379"
    etcd_client_port: int = 2379
    etcd_peer_port: int = 2380
    etcd_data_dir: str | None = None


class CoordinatorConfig(BaseModel):
    """Coordinator HTTP API settings."""

    host: str = "127.0.0.1"
    port: int = 8080

    @property
    def url(self) -> str:
        """Return the coordinator base URL."""
        return f"http://{self.host}:{self.port}"


class SchedulerConfig(BaseModel):
    """Per-term weights for the paper-style scheduler cost function.

    Each weight scales one signal that contributes to a worker's cost; the
    scheduler picks the lowest-cost worker. The base 5 weights are bounded
    to ``[0.0, 1.0]``; ``nu`` (RouteLLM quality term) is bounded to
    ``[0.0, 20.0]``, matching the operating points in SCHEDULER.md
    (``nu=5`` and ``nu=20``).

    The six weights map to the cost terms in ``src/coordinator/scheduler.py``:

    * ``queue``       — punishes deep queues (relative to decode speed).
    * ``prefix_miss`` — punishes prefix-cache misses (rewards cache hits).
    * ``memory``      — punishes KV/RAM pressure on the worker.
    * ``jitter``      — punishes flaky network paths.
    * ``thermal``     — punishes thermally-throttled workers.
    * ``nu``          — quality-aware routing knob: ``nu * complexity *
      (1 - quality)``. ``0`` disables it.
    """

    queue: float = Field(default=1.0, ge=0.0, le=1.0)
    prefix_miss: float = Field(default=1.0, ge=0.0, le=1.0)
    memory: float = Field(default=1.0, ge=0.0, le=1.0)
    jitter: float = Field(default=1.0, ge=0.0, le=1.0)
    thermal: float = Field(default=1.0, ge=0.0, le=1.0)
    nu: float = Field(default=0.0, ge=0.0, le=20.0)


KNOWN_MODALITIES: tuple[str, ...] = ("text", "image", "audio", "video")


class WorkerConfig(BaseModel):
    """Worker HTTP API settings."""

    host: str = "127.0.0.1"
    port: int = 9090
    coordinator_url: str | None = None
    weight: float = Field(default=0.0, ge=-1.0, le=1.0)
    model_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    modalities: list[str] = Field(default_factory=lambda: ["text"])


class EngineConfig(BaseModel):
    """Inference engine settings for this node."""

    name: EngineName = "llama.cpp"
    server_bin: str = "llama-server"
    server_host: str = "127.0.0.1"
    server_port: int = 9081
    extra_args: str = ""


class ModelConfig(BaseModel):
    """Model available on this node."""

    id: str
    engine: EngineName = "llama.cpp"
    path: str
    source: str = "local"
    repo_id: str | None = None
    filename: str | None = None
    quant: str | None = None
    context_length: int | None = None
    loaded: bool = True


class RelayConfig(BaseModel):
    """Complete local Relay runtime config."""

    version: int = 1
    node_id: str = Field(default_factory=lambda: socket.gethostname() or "relay-node")
    cluster_id: str = Field(default_factory=lambda: f"relay-{uuid.uuid4().hex[:8]}")
    role: NodeRole = "dual"
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    membership: MembershipConfig = Field(default_factory=MembershipConfig)
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    models: list[ModelConfig] = Field(default_factory=list)

    @property
    def runs_coordinator(self) -> bool:
        """Return whether this node starts a coordinator process."""
        return self.role in {"coordinator", "dual"}

    @property
    def runs_worker(self) -> bool:
        """Return whether this node starts a worker process."""
        return self.role in {"worker", "dual"}

    def active_model(self) -> ModelConfig | None:
        """Return the first loaded model for the node, if any."""
        for model in self.models:
            if model.loaded:
                return model
        return self.models[0] if self.models else None

    def with_model(self, model: ModelConfig) -> RelayConfig:
        """Return a config with ``model`` inserted or replacing the same id."""
        models = [existing for existing in self.models if existing.id != model.id]
        models.insert(0, model)
        return self.model_copy(update={"models": models})


def load_config(path: Path | None = None) -> RelayConfig:
    """Load Relay config from disk."""
    config_path = path or RelayPaths.from_home().config
    if not config_path.exists():
        raise ConfigError(f"Relay config not found at {config_path}. Run `relay init` first.")
    return RelayConfig.model_validate_json(config_path.read_text())


def save_config(config: RelayConfig, path: Path | None = None) -> None:
    """Write Relay config to disk."""
    paths = RelayPaths.from_home()
    paths.ensure()
    config_path = path or paths.config
    config_path.write_text(config.model_dump_json(indent=2) + "\n")


def config_exists(path: Path | None = None) -> bool:
    """Return whether a Relay config file exists."""
    return (path or RelayPaths.from_home().config).exists()


def coordinator_host_from_url(url: str) -> str:
    """Extract host from a coordinator URL or host string."""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    if not parsed.hostname:
        raise ConfigError(f"Invalid coordinator URL: {url}")
    return parsed.hostname


def normalize_coordinator_url(value: str) -> str:
    """Normalize coordinator input to a base HTTP URL."""
    parsed = urlparse(value if "://" in value else f"http://{value}")
    if not parsed.hostname:
        raise ConfigError(f"Invalid coordinator URL: {value}")
    port = parsed.port or 8080
    return f"{parsed.scheme}://{parsed.hostname}:{port}"


def model_inventory_json(config: RelayConfig) -> str:
    """Return JSON model inventory published by the worker."""
    inventory = {
        "engines": [config.engine.name] if config.runs_worker else [],
        "models": [model.model_dump() for model in config.models if model.loaded],
    }
    return json.dumps(inventory)
