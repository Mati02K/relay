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
    EngineName,
    MembershipConfig,
    ModelConfig,
    NetworkBackend,
    NetworkConfig,
    NodeRole,
    RelayConfig,
    SchedulerConfig,
    WorkerConfig,
    coordinator_host_from_url,
    normalize_coordinator_url,
    save_config,
)
from relay.models import (
    MLX_MODEL_CATALOG,
    MODEL_CATALOG,
    find_mlx_catalog_model,
    pull_catalog_model,
    pull_mlx_model,
    register_local_model,
)
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
    worker_weight: float | None = None
    engine: str | None = None


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
    worker_weight = _select_worker_weight(role, options.worker_weight)
    engine_name = _select_engine(role, options.engine)
    config_kwargs: dict[str, object] = {
        "role": role,
        "network": NetworkConfig(backend=network),
        "membership": MembershipConfig(host=membership_host),
        "coordinator": CoordinatorConfig(host=host),
        "worker": WorkerConfig(
            host=host,
            coordinator_url=coordinator_url,
            weight=worker_weight,
        ),
        "engine": EngineConfig(name=engine_name),
    }
    if prompted_node_id:
        config_kwargs["node_id"] = prompted_node_id
    config = RelayConfig.model_validate(config_kwargs)

    if config.runs_worker and not options.skip_model:
        if engine_name == "mlx":
            config = _configure_model_mlx(config, options)
        else:
            config = _configure_model(config, options)

    if config.runs_coordinator:
        config = _configure_scheduler(config)

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


def _select_engine(role: NodeRole, flag_value: str | None) -> EngineName:
    """Resolve inference engine from flag or interactive prompt.

    Coordinator-only nodes don't run inference; return the default without
    prompting. For worker/dual roles ask the user to choose.
    """
    if role == "coordinator":
        return "llama.cpp"
    if flag_value:
        return _validate_choice(flag_value, ("llama.cpp", "mlx"), "engine")  # type: ignore[return-value]
    return _prompt_choice(  # type: ignore[return-value]
        "Inference engine",
        ("llama.cpp", "mlx"),
        default="llama.cpp",
        suffix={
            "llama.cpp": "local GGUF model via llama-server",
            "mlx": "HuggingFace model via mlx_lm.server (Apple Silicon)",
        },
    )


def _configure_model_mlx(config: RelayConfig, options: InitOptions) -> RelayConfig:
    """Configure an MLX model, reusing already-downloaded models when available.

    Mirrors the llamacpp flow: if models exist in ~/.relay/models/ the user is
    offered a local-pick menu first. They can also pull a new model from
    HuggingFace or skip and configure later.
    """
    if options.model_path:
        model_id = options.model or Path(options.model_path).name
        model = ModelConfig(
            id=model_id,
            engine="mlx",
            path=options.model_path,
            source="local",
            loaded=True,
        )
        return config.with_model(model)

    if options.model:
        repo_id = _resolve_mlx_model_selection(options.model)
        updated, _ = pull_mlx_model(config, repo_id)
        return updated

    action = _prompt_choice(
        "MLX model setup",
        ("local", "pull", "skip"),
        default="skip",
        suffix={
            "local": "use an already-downloaded MLX model from ~/.relay/models/",
            "pull": "download a new model from HuggingFace",
            "skip": "configure later with relay init",
        },
    )
    if action == "skip":
        return config
    if action == "local":
        return _select_local_mlx_model(config)

    print("Available MLX models:")
    for index, m in enumerate(MLX_MODEL_CATALOG, start=1):
        print(f"  {index}. {m.id} — {m.label}")
    raw = _prompt_text(
        "Model id, number, or HuggingFace repo (e.g. mlx-community/...)",
        default=None,
        required=False,
    )
    if not raw:
        return config
    repo_id = _resolve_mlx_model_selection(raw)
    updated, _ = pull_mlx_model(config, repo_id)
    return updated


def _select_local_mlx_model(config: RelayConfig) -> RelayConfig:
    """Pick an already-downloaded MLX model from ~/.relay/models/."""
    paths = RelayPaths.from_home()
    found = _discover_local_mlx_models(paths.models)
    if not found:
        print(
            "No MLX models found in",
            paths.models,
            "— use 'pull' to download one first.",
        )
        return config
    print("Local MLX models found:")
    for index, (label, path) in enumerate(found, start=1):
        print(f"  {index}. {label}")
    raw = _prompt_text("Pick a model number", default="1", required=False)
    chosen_path = _resolve_local_pick(raw, found)
    model = ModelConfig(
        id=chosen_path.name,
        engine="mlx",
        path=str(chosen_path),
        source="local",
        loaded=True,
    )
    return config.with_model(model)


def _discover_local_mlx_models(models_root: Path) -> list[tuple[str, Path]]:
    """Return (label, path) pairs for every MLX model dir under the Relay models dir.

    A directory is considered an MLX model if it contains config.json plus at
    least one .safetensors or .npz weight file.
    """
    if not models_root.is_dir():
        return []
    discovered: list[tuple[str, Path]] = []
    for d in sorted(models_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "config.json").exists():
            continue
        has_weights = any(d.glob("*.safetensors")) or any(d.glob("*.npz"))
        if not has_weights:
            continue
        size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024**2)
        label = f"{d.name} ({size_mb:.0f} MB)"
        discovered.append((label, d))
    return discovered


def _resolve_mlx_model_selection(value: str) -> str:
    """Return a HuggingFace repo ID from a catalog id, number, or raw repo ID."""
    entry = find_mlx_catalog_model(value)
    if entry:
        return entry.repo_id
    return value.strip()


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
        return _select_local_model(config)

    print("Available models:")
    for index, model in enumerate(MODEL_CATALOG, start=1):
        print(f"  {index}. {model.id} ({model.label})")
    selected = _prompt_text("Model id or number", default=MODEL_CATALOG[1].id, required=False)
    selected = _resolve_catalog_model_selection(selected)
    updated, _ = pull_catalog_model(config, selected)
    return updated


def _configure_scheduler(config: RelayConfig) -> RelayConfig:
    """Ask whether to keep default scheduler weights or tune each one.

    Only runs for coordinator/dual nodes — workers don't schedule anyone.
    Default keeps all five weights at 1.0 (paper-style equal influence).
    Advanced asks one prompt per weight, defaulting to the current value.
    """
    action = _prompt_choice(
        "Scheduler tuning",
        ("default", "advanced"),
        default="default",
        suffix={
            "default": "all weights = 1.0 (recommended)",
            "advanced": "tune queue / prefix / memory / jitter / thermal",
        },
    )
    if action == "default":
        return config

    print(
        "Pick a weight per signal in the range [0.0, 1.0]. "
        "1.0 = full influence, 0.0 = signal disabled, in-between dampens it."
    )
    print("Enter to keep the current value.")
    current = config.scheduler
    weights = SchedulerConfig(
        queue=_prompt_float(
            "Queue weight (deep queue vs. decode speed)",
            default=current.queue,
        ),
        prefix_miss=_prompt_float(
            "Prefix-miss weight (rewards cache hits)",
            default=current.prefix_miss,
        ),
        memory=_prompt_float(
            "Memory weight (KV/RAM pressure)",
            default=current.memory,
        ),
        jitter=_prompt_float(
            "Jitter weight (flaky network paths)",
            default=current.jitter,
        ),
        thermal=_prompt_float(
            "Thermal weight (throttled workers)",
            default=current.thermal,
        ),
    )
    return config.model_copy(update={"scheduler": weights})


def _prompt_float(label: str, *, default: float) -> float:
    """Prompt for a weight in ``[0.0, 1.0]``; blank input keeps ``default``."""
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print(f"Invalid number: {raw!r}. Try again.")
            continue
        if value < 0.0 or value > 1.0:
            print(f"Weight must be between 0.0 and 1.0 (got {value}). Try again.")
            continue
        return value


def _select_worker_weight(role: NodeRole, flag_value: float | None) -> float:
    """Resolve worker-weight from the CLI flag or interactive prompt.

    Only meaningful for nodes that run a worker. Coordinator-only nodes skip
    the prompt and store the default 0.0 (unused by the scheduler).
    """
    if role == "coordinator":
        return 0.0
    if flag_value is not None:
        if not -1.0 <= flag_value <= 1.0:
            raise ConfigError(
                f"worker-weight must be between -1.0 and 1.0 (got {flag_value})"
            )
        return flag_value
    while True:
        raw = input("Worker weight (scheduler preference) [0.0]: ").strip()
        if not raw:
            return 0.0
        try:
            value = float(raw)
        except ValueError:
            print(f"Invalid number: {raw!r}. Try again.")
            continue
        if value < -1.0 or value > 1.0:
            print(f"Worker weight must be between -1.0 and 1.0 (got {value}). Try again.")
            continue
        return value


def _select_local_model(config: RelayConfig) -> RelayConfig:
    """Pick a GGUF from ``~/.relay/models/`` by number, no path typing required.

    Falls back to asking for an absolute path only when no downloaded GGUFs
    are found (e.g. nothing pulled yet).
    """
    paths = RelayPaths.from_home()
    found = _discover_local_ggufs(paths.models)
    if not found:
        print(
            "No GGUF files found in",
            paths.models,
            "— pulling one is usually easier than typing a path.",
        )
        raw_path = _prompt_text("Local GGUF path", required=True)
        chosen = Path(raw_path).expanduser()
        return register_local_model(config, chosen.stem, chosen)

    print("Local GGUFs found:")
    for index, (label, path) in enumerate(found, start=1):
        print(f"  {index}. {label}")
    default_index = "1"
    raw = _prompt_text(
        "Pick a model number",
        default=default_index,
        required=False,
    )
    chosen_path = _resolve_local_pick(raw, found)
    # Model id = filename stem. Skip the extra prompt — users picked from a
    # menu, no need for a second decision. Rename via config.json if needed.
    return register_local_model(config, chosen_path.stem, chosen_path)


def _discover_local_ggufs(models_root: Path) -> list[tuple[str, Path]]:
    """Return ``(label, path)`` pairs for every GGUF under the Relay models dir.

    Sorted alphabetically by label so menu numbering is stable across runs.
    """
    if not models_root.is_dir():
        return []
    discovered: list[tuple[str, Path]] = []
    for gguf in sorted(models_root.rglob("*.gguf")):
        try:
            rel = gguf.relative_to(models_root)
        except ValueError:
            rel = Path(gguf.name)
        size_gb = gguf.stat().st_size / (1024**3)
        label = f"{rel} ({size_gb:.1f} GB)"
        discovered.append((label, gguf))
    return discovered


def _resolve_local_pick(raw: str, found: list[tuple[str, Path]]) -> Path:
    """Resolve a menu number into the chosen path; raise on invalid input."""
    value = raw.strip()
    if not value:
        return found[0][1]
    if value.isdigit():
        idx = int(value)
        if 1 <= idx <= len(found):
            return found[idx - 1][1]
    raise ConfigError(f"Invalid selection '{raw}'. Pick a number between 1 and {len(found)}.")


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
