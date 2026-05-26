"""Interactive and flag-driven ``relay init`` implementation."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from relay.config import (
    ConfigError,
    CoordinatorConfig,
    EngineConfig,
    MembershipConfig,
    NetworkBackend,
    NetworkConfig,
    NodeRole,
    RelayConfig,
    WorkerConfig,
    coordinator_host_from_url,
    normalize_coordinator_url,
    save_config,
)
from relay.models import MODEL_CATALOG, pull_catalog_model, register_local_model
from relay.paths import RelayPaths
from relay.software import ensure_runtime_software
from relay.supervisor import stop as stop_managed_processes
from relay.ui import muted, warn


@dataclass(frozen=True)
class InitOptions:
    """Inputs accepted by ``relay init``."""

    role: str | None
    network: str | None
    coordinator: str | None
    node_id: str | None
    host: str | None
    model: str | None
    model_path: str | None
    skip_model: bool


def run_init(options: InitOptions) -> RelayConfig:
    """Create and persist a Relay config, overwriting any existing config.

    Always stops Relay processes started by a previous ``relay start`` and
    wipes the etcd data directory. This keeps re-running ``relay init`` safe:
    no zombie processes against a stale config, and the freshly-generated
    ``cluster_id`` matches the (now empty) etcd state directory so the next
    ``relay start`` actually boots.
    """
    paths = RelayPaths.from_home()
    paths.ensure()
    _reset_runtime_state(paths)

    role = _select_role(options.role)
    network = _select_network(options.network)
    host = options.host or _default_host(network)

    coordinator_url: str | None = None
    if role == "worker":
        raw = options.coordinator or _prompt_text("Coordinator URL or IP", required=True)
        coordinator_url = normalize_coordinator_url(raw)
        membership_host = coordinator_host_from_url(coordinator_url)
    else:
        membership_host = host

    prompted_node_id = options.node_id or _prompt_text("Node id", default=None, required=False)
    config_kwargs: dict[str, object] = {
        "role": role,
        "network": NetworkConfig(backend=network),
        "membership": MembershipConfig(host=membership_host),
        "coordinator": CoordinatorConfig(host=host),
        "worker": WorkerConfig(host=host, coordinator_url=coordinator_url),
        "engine": EngineConfig(),
    }
    if prompted_node_id:
        config_kwargs["node_id"] = prompted_node_id
    config = RelayConfig.model_validate(config_kwargs)

    if config.runs_worker and not options.skip_model:
        config = _configure_model(config, options)

    print("Preparing runtime software...")
    ensure_runtime_software(config)
    save_config(config)
    return config


def _reset_runtime_state(paths: RelayPaths) -> None:
    """Stop any running Relay processes and clear the etcd data dir."""
    statuses = stop_managed_processes(config=None)
    stopped_names = [s.name for s in statuses if s.detail == "stopped"]
    if stopped_names:
        print(warn("stopped:"), ", ".join(stopped_names))
    if paths.etcd_data.exists():
        shutil.rmtree(paths.etcd_data)
        print(muted("cleared:"), str(paths.etcd_data))


def _select_role(value: str | None) -> NodeRole:
    if value:
        return cast(
            NodeRole,
            _validate_choice(value, ("coordinator", "worker", "dual"), "role"),
        )
    return cast(
        NodeRole,
        _prompt_choice(
            "Node role",
            ("coordinator", "worker", "dual"),
            default="dual",
            suffix={"dual": "coordinator + worker on this machine"},
        ),
    )


def _select_network(value: str | None) -> NetworkBackend:
    if value:
        return cast(
            NetworkBackend,
            _validate_choice(value, ("tailscale", "lan"), "network"),
        )
    return cast(
        NetworkBackend,
        _prompt_choice(
            "Network backend",
            ("tailscale", "lan"),
            default="tailscale",
            suffix={"lan": "future/experimental"},
        ),
    )


def _configure_model(config: RelayConfig, options: InitOptions) -> RelayConfig:
    if options.model_path:
        model_id = options.model or Path(options.model_path).stem
        return register_local_model(config, model_id, Path(options.model_path))
    if options.model:
        updated, _ = pull_catalog_model(config, _resolve_catalog_model_selection(options.model))
        return updated

    action = _prompt_choice(
        "Model setup",
        ("pull", "local", "skip"),
        default="skip",
        suffix={
            "pull": "download GGUF from Hugging Face",
            "local": "use an existing GGUF file",
            "skip": "configure later with relay pull",
        },
    )
    if action == "skip":
        return config
    if action == "local":
        path = _prompt_text("Local GGUF path", required=True)
        model_id = _prompt_text("Model id", default=Path(path).stem, required=False)
        return register_local_model(config, model_id, Path(path))

    print("Available models:")
    for index, model in enumerate(MODEL_CATALOG, start=1):
        print(f"  {index}. {model.id} ({model.label})")
    selected = _prompt_text("Model id or number", default=MODEL_CATALOG[1].id, required=False)
    selected = _resolve_catalog_model_selection(selected)
    updated, _ = pull_catalog_model(config, selected)
    return updated


def _prompt_choice(
    label: str,
    choices: tuple[str, ...],
    *,
    default: str,
    suffix: dict[str, str] | None = None,
) -> str:
    suffix = suffix or {}
    print(label + ":")
    for index, choice in enumerate(choices, start=1):
        extra = f" ({suffix[choice]})" if choice in suffix else ""
        default_marker = " [default]" if choice == default else ""
        print(f"  {index}. {choice}{extra}{default_marker}")
    raw = input("> ").strip()
    if not raw:
        return default
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(choices):
            return choices[idx - 1]
    return _validate_choice(raw, choices, label)


def _prompt_text(label: str, *, default: str | None = None, required: bool) -> str:
    prompt = f"{label}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    while True:
        value = input(prompt).strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("Value is required.")


def _validate_choice(value: str, choices: tuple[str, ...], label: str) -> str:
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ConfigError(f"Invalid {label}: {value}. Expected one of: {', '.join(choices)}")
    return normalized


def _resolve_catalog_model_selection(value: str) -> str:
    """Return a catalog model id from either a model id or a 1-based menu number."""
    selected = value.strip().lower()
    if selected.isdigit():
        index = int(selected)
        if 1 <= index <= len(MODEL_CATALOG):
            return MODEL_CATALOG[index - 1].id
        raise ConfigError(f"Model number must be between 1 and {len(MODEL_CATALOG)}")
    return selected


def _default_host(network: NetworkBackend) -> str:
    if network != "tailscale":
        return "127.0.0.1"
    tailscale = shutil.which("tailscale")
    if not tailscale:
        return "127.0.0.1"
    try:
        result = subprocess.run(
            [tailscale, "ip", "-4"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "127.0.0.1"
    address = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    return address or "127.0.0.1"
