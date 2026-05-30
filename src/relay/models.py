"""Model catalog and Hugging Face download helpers."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from relay.config import ModelConfig, RelayConfig
from relay.paths import RelayPaths


@dataclass(frozen=True)
class CatalogModel:
    """A downloadable GGUF model option exposed by the CLI."""

    id: str
    label: str
    repo_id: str
    file_patterns: tuple[str, ...]
    quant: str = "Q4_K_M"
    context_length: int | None = None


MODEL_CATALOG: tuple[CatalogModel, ...] = (
    CatalogModel(
        id="qwen2.5-0.5b",
        label="Qwen2.5 0.5B Instruct",
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        file_patterns=("*q4_k_m*.gguf", "*q4_0*.gguf", "*.gguf"),
    ),
    CatalogModel(
        id="qwen2.5-1.5b",
        label="Qwen2.5 1.5B Instruct",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        file_patterns=("*q4_k_m*.gguf", "*q4_0*.gguf", "*.gguf"),
    ),
    CatalogModel(
        id="qwen2.5-3b",
        label="Qwen2.5 3B Instruct",
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        file_patterns=("*q4_k_m*.gguf", "*q4_0*.gguf", "*.gguf"),
    ),
    CatalogModel(
        id="llama-3.2-1b",
        label="Llama 3.2 1B Instruct",
        repo_id="gnec/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
        file_patterns=("*q4_k_m*.gguf", "*.gguf"),
    ),
    CatalogModel(
        id="llama-3.2-3b",
        label="Llama 3.2 3B Instruct",
        repo_id="unsloth/Llama-3.2-3B-Instruct-GGUF",
        file_patterns=("*q4_k_m*.gguf", "*q4_0*.gguf", "*.gguf"),
    ),
    CatalogModel(
        id="phi-3.5-mini",
        label="Phi 3.5 Mini Instruct",
        repo_id="bartowski/Phi-3.5-mini-instruct-GGUF",
        file_patterns=("*q4_k_m*.gguf", "*q4_0*.gguf", "*.gguf"),
    ),
    CatalogModel(
        id="mistral-7b",
        label="Mistral 7B Instruct v0.3",
        repo_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        file_patterns=("*q4_k_m*.gguf", "*q4_0*.gguf", "*.gguf"),
    ),
    CatalogModel(
        id="qwen2.5-coder-7b",
        label="Qwen2.5 Coder 7B Instruct",
        # bartowski's repo ships a single-file Q4_K_M; the official
        # Qwen/Qwen2.5-Coder-7B-Instruct-GGUF splits it across 4 parts which
        # our downloader cannot reassemble.
        repo_id="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        file_patterns=("*q4_k_m*.gguf", "*q4_0*.gguf", "*.gguf"),
    ),
    CatalogModel(
        id="qwen-moe-a2.7b",
        label="Qwen1.5 MoE A2.7B Chat (14B total, 2.7B active)",
        # Note the unusual repo naming: RichardErkhov mirrors HF repos by
        # replacing the slash with "_-_" and lowercasing "gguf".
        repo_id="RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf",
        file_patterns=("*q4_k_m*.gguf", "*q4_0*.gguf", "*.gguf"),
    ),
)


@dataclass(frozen=True)
class MlxCatalogModel:
    """A downloadable MLX model option exposed by the CLI."""

    id: str
    label: str
    repo_id: str


MLX_MODEL_CATALOG: tuple[MlxCatalogModel, ...] = (
    MlxCatalogModel(
        id="qwen2.5-0.5b-mlx",
        label="Qwen2.5 0.5B Instruct (MLX 4-bit) ~300 MB",
        repo_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    ),
    MlxCatalogModel(
        id="qwen2.5-1.5b-mlx",
        label="Qwen2.5 1.5B Instruct (MLX 4-bit) ~900 MB",
        repo_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    ),
    MlxCatalogModel(
        id="llama-3.2-1b-mlx",
        label="Llama 3.2 1B Instruct (MLX 4-bit) ~600 MB",
        repo_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
    ),
)


class ModelCatalogError(RuntimeError):
    """Raised when a requested model cannot be resolved."""


def find_catalog_model(model_id: str) -> CatalogModel:
    """Return a catalog model by id."""
    normalized = model_id.strip().lower()
    for model in MODEL_CATALOG:
        if model.id == normalized:
            return model
    available = ", ".join(model.id for model in MODEL_CATALOG)
    raise ModelCatalogError(f"Unknown model '{model_id}'. Available: {available}")


def catalog_rows() -> list[str]:
    """Return display rows for the model catalog."""
    return [f"{model.id:18} {model.label} ({model.repo_id})" for model in MODEL_CATALOG]


def register_local_model(config: RelayConfig, model_id: str, path: Path) -> RelayConfig:
    """Add a local GGUF model path to config."""
    resolved = path.expanduser()
    model = ModelConfig(
        id=model_id,
        engine=config.engine.name,
        path=str(resolved),
        source="local",
        filename=resolved.name,
    )
    return config.with_model(model)


def pull_catalog_model(config: RelayConfig, model_id: str) -> tuple[RelayConfig, ModelConfig]:
    """Download a catalog GGUF model from Hugging Face and add it to config."""
    catalog_model = find_catalog_model(model_id)
    hf = _load_huggingface_hub()
    filename = _select_gguf_filename(
        list_repo_files=hf["list_repo_files"],
        repo_id=catalog_model.repo_id,
        patterns=catalog_model.file_patterns,
    )

    paths = RelayPaths.from_home()
    paths.ensure()
    print(f"Downloading {filename} from {catalog_model.repo_id}...")
    try:
        local_path = hf["hf_hub_download"](
            repo_id=catalog_model.repo_id,
            filename=filename,
            local_dir=str(paths.models / catalog_model.id),
        )
    except Exception as e:
        err = str(e)
        if "401" in err or "credentials" in err.lower() or "token" in err.lower():
            raise ModelCatalogError(
                f"HuggingFace authentication required to download {catalog_model.repo_id}.\n"
                "Log in with:  huggingface-cli login\n"
                "Then re-run relay init or relay pull."
            ) from e
        raise ModelCatalogError(
            f"Failed to download {filename} from {catalog_model.repo_id}: {e}"
        ) from e
    model = ModelConfig(
        id=catalog_model.id,
        engine=config.engine.name,
        path=str(local_path),
        source="huggingface",
        repo_id=catalog_model.repo_id,
        filename=filename,
        quant=catalog_model.quant,
        context_length=catalog_model.context_length,
        loaded=True,
    )
    return config.with_model(model), model


def pull_mlx_model(config: RelayConfig, repo_id: str) -> tuple[RelayConfig, ModelConfig]:
    """Download an MLX model repo from HuggingFace into ~/.relay/models/.

    Uses ``snapshot_download`` with ``local_dir`` so files land in the same
    place as GGUF models rather than the HuggingFace cache directory.
    """
    hf = _load_huggingface_hub()
    model_id = repo_id.split("/")[-1]
    paths = RelayPaths.from_home()
    paths.ensure()
    local_dir = paths.models / model_id
    print(f"Downloading {repo_id} to {local_dir} (this may take a few minutes)...")
    try:
        local_path = hf["snapshot_download"](repo_id=repo_id, local_dir=str(local_dir))
    except Exception as e:
        err = str(e)
        if "401" in err or "credentials" in err.lower() or "token" in err.lower():
            raise ModelCatalogError(
                f"HuggingFace authentication required to download {repo_id}.\n"
                "Log in with:  huggingface-cli login\n"
                "Then re-run relay init."
            ) from e
        if "404" in err or "Repository Not Found" in err:
            raise ModelCatalogError(
                f"Model not found: {repo_id}\n"
                "Check the repo ID at https://huggingface.co/mlx-community"
            ) from e
        raise ModelCatalogError(f"Failed to download {repo_id}: {e}") from e
    model = ModelConfig(
        id=model_id,
        engine="mlx",
        path=str(local_path),
        source="huggingface",
        repo_id=repo_id,
        loaded=True,
    )
    print(f"Downloaded {model_id}: {local_path}")
    return config.with_model(model), model


def mlx_catalog_rows() -> list[str]:
    """Return display rows for the MLX model catalog."""
    return [f"{m.id:22} {m.label} ({m.repo_id})" for m in MLX_MODEL_CATALOG]


def find_mlx_catalog_model(value: str) -> MlxCatalogModel | None:
    """Resolve a catalog id or 1-based number to an MLX catalog entry."""
    stripped = value.strip()
    if stripped.isdigit():
        idx = int(stripped)
        if 1 <= idx <= len(MLX_MODEL_CATALOG):
            return MLX_MODEL_CATALOG[idx - 1]
        return None
    for m in MLX_MODEL_CATALOG:
        if m.id == stripped.lower():
            return m
    return None


def _load_huggingface_hub() -> dict[str, Any]:
    try:
        module = import_module("huggingface_hub")
    except ImportError as e:
        raise ModelCatalogError(
            "huggingface_hub is required for `relay pull`. "
            "Install the package dependencies, then retry."
        ) from e
    return {
        "hf_hub_download": getattr(module, "hf_hub_download"),
        "list_repo_files": getattr(module, "list_repo_files"),
        "snapshot_download": getattr(module, "snapshot_download"),
    }


def _select_gguf_filename(
    *,
    list_repo_files: Any,
    repo_id: str,
    patterns: tuple[str, ...],
) -> str:
    try:
        files = [str(item) for item in list_repo_files(repo_id=repo_id)]
    except Exception as e:
        raise ModelCatalogError(f"Failed to list GGUF files in {repo_id}: {e}") from e
    gguf_files = [item for item in files if item.endswith(".gguf")]
    for pattern in patterns:
        matches = sorted(
            item for item in gguf_files if fnmatch.fnmatch(item.lower(), pattern.lower())
        )
        if matches:
            return matches[0]
    raise ModelCatalogError(f"No GGUF file found in {repo_id}")
