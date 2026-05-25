"""Software and environment checks for the Relay CLI."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from relay.config import RelayConfig
from relay.paths import RelayPaths

LLAMA_CPP_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
GO_RELEASE_API = "https://go.dev/dl/?mode=json"
MIN_GO_VERSION = (1, 24)
ETCD_VERSION = "v3.5.14"


@dataclass(frozen=True)
class CheckResult:
    """One doctor check result."""

    name: str
    ok: bool
    detail: str


def resolve_executable(configured: str | None, fallback_name: str) -> str | None:
    """Resolve an executable from a configured path or PATH lookup."""
    use_system_path = os.getenv("RELAY_SKIP_SYSTEM_BINARIES") != "1"
    if configured:
        path = Path(configured).expanduser()
        if path.exists():
            return str(path)
        found = shutil.which(configured) if use_system_path else None
        if found is not None:
            return found
        if configured == fallback_name:
            return _common_executable(fallback_name)
        return None
    return (shutil.which(fallback_name) if use_system_path else None) or _common_executable(
        fallback_name
    )


def ensure_runtime_software(config: RelayConfig) -> None:
    """Install or build runtime software required by the selected node role."""
    if config.runs_coordinator:
        ensure_etcd()
        ensure_membership_etcd()
    if config.runs_worker:
        ensure_llama_server()


def ensure_etcd() -> str:
    """Return an etcd binary path, downloading the managed release when missing."""
    existing = resolve_executable("etcd", "etcd")
    if existing:
        return existing

    paths = RelayPaths.from_home()
    paths.ensure()
    archive_name = _etcd_archive_name()
    archive_path = paths.cache / archive_name
    install_dir = paths.bin / ETCD_VERSION
    etcd_path = _find_named_binary(install_dir, _binary_name("etcd"))
    if etcd_path is not None:
        return str(etcd_path)

    if not archive_path.exists():
        _download_file(_etcd_download_url(archive_name), archive_path)
    _extract_archive(archive_path, install_dir)
    etcd_path = _find_named_binary(install_dir, _binary_name("etcd"))
    if etcd_path is None:
        raise RuntimeError(f"Downloaded etcd archive did not contain etcd: {archive_name}")
    etcd_path.chmod(etcd_path.stat().st_mode | 0o111)
    return str(etcd_path)


def ensure_membership_etcd() -> str:
    """Return a built membership-etcd binary, building it with managed Go when missing."""
    existing = resolve_executable("membership-etcd", "membership-etcd")
    if existing:
        return existing

    paths = RelayPaths.from_home()
    paths.ensure()
    target = paths.bin / _binary_name("membership-etcd")
    source_dir = _membership_etcd_source_dir()
    if not source_dir.exists():
        raise RuntimeError(f"membership-etcd source not found: {source_dir}")
    build_dir = paths.cache / "membership-etcd-src"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(source_dir, build_dir)

    go_bin = ensure_go()
    env = os.environ.copy()
    env["GOMODCACHE"] = str(paths.cache / "go-mod")
    env["GOCACHE"] = str(paths.cache / "go-build")
    env["GOTOOLCHAIN"] = "local"
    download = subprocess.run(
        [go_bin, "mod", "download"],
        cwd=build_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if download.returncode != 0:
        detail = "\n".join(part for part in (download.stdout, download.stderr) if part)
        raise RuntimeError(f"Failed to download membership-etcd Go modules:\n{detail}")
    result = subprocess.run(
        [go_bin, "build", "-buildvcs=false", "-o", str(target), "."],
        cwd=build_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        detail = "\n".join(part for part in (result.stdout, result.stderr) if part)
        raise RuntimeError(f"Failed to build membership-etcd:\n{detail}")
    target.chmod(target.stat().st_mode | 0o111)
    return str(target)


def _membership_etcd_source_dir() -> Path:
    current = Path(__file__).resolve()
    candidates = [
        current.parents[1] / "membership" / "etcd-go",
        current.parents[2] / "src" / "membership" / "etcd-go",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def ensure_go() -> str:
    """Return a Go toolchain path, downloading a managed toolchain when needed."""
    system_go = shutil.which("go") if os.getenv("RELAY_SKIP_SYSTEM_BINARIES") != "1" else None
    if system_go and _go_version_ok(system_go):
        return system_go

    paths = RelayPaths.from_home()
    paths.ensure()
    release = _select_go_release()
    version = str(release["version"])
    archive_name = str(release["filename"])
    archive_path = paths.cache / archive_name
    install_dir = paths.bin / version
    go_bin = install_dir / "go" / "bin" / _binary_name("go")
    if go_bin.exists():
        return str(go_bin)

    if not archive_path.exists():
        _download_file(str(release["url"]), archive_path)
    _extract_archive(archive_path, install_dir)
    if not go_bin.exists():
        raise RuntimeError(f"Downloaded Go archive did not contain go binary: {archive_name}")
    go_bin.chmod(go_bin.stat().st_mode | 0o111)
    return str(go_bin)


def ensure_llama_server() -> str:
    """Return a usable llama-server path, downloading a prebuilt binary when missing."""
    existing = resolve_executable("llama-server", "llama-server")
    if existing:
        return existing

    paths = RelayPaths.from_home()
    paths.ensure()
    release = _fetch_latest_llama_cpp_release()
    tag = str(release["tag_name"])
    asset = _select_llama_cpp_asset(release)
    archive_path = paths.cache / asset["name"]
    install_dir = paths.bin / f"llama.cpp-{tag}"
    installed = _find_llama_server(install_dir)
    if installed is not None:
        return str(installed)

    if not archive_path.exists():
        _download_file(str(asset["browser_download_url"]), archive_path)
    _extract_archive(archive_path, install_dir)
    found = _find_llama_server(install_dir)
    if found is None:
        raise RuntimeError(
            f"Downloaded llama.cpp archive did not contain llama-server: {asset['name']}"
        )
    found.chmod(found.stat().st_mode | 0o111)
    return str(found)


def _common_executable(name: str) -> str | None:
    candidates: list[Path] = []
    paths = RelayPaths.from_home()
    if name == "llama-server":
        candidates.append(paths.bin / "llama-server")
        candidates.extend(sorted(paths.bin.glob("llama.cpp-*/**/llama-server")))
        if os.getenv("RELAY_SKIP_TMP_LLAMA_SERVER") != "1":
            candidates.append(Path("/tmp/relay-llama-build/bin/llama-server"))
    elif name == "etcd":
        candidates.append(paths.bin / "etcd")
        candidates.extend(sorted(paths.bin.glob("v*/**/etcd")))
    elif name == "membership-etcd":
        candidates.append(paths.bin / "membership-etcd")
    elif name == "go":
        candidates.extend(sorted(paths.bin.glob("go*/go/bin/go")))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _binary_name(name: str) -> str:
    return f"{name}.exe" if platform.system().lower() == "windows" else name


def _fetch_latest_llama_cpp_release() -> dict[str, Any]:
    with urllib.request.urlopen(LLAMA_CPP_RELEASE_API, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict) or not data.get("tag_name"):
        raise RuntimeError("Could not read latest llama.cpp release metadata")
    return data


def _select_llama_cpp_asset(release: dict[str, Any]) -> dict[str, str]:
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("llama.cpp release metadata has no assets")
    wanted = _llama_cpp_asset_fragment()
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if wanted in name and url:
            return {"name": name, "browser_download_url": url}
    raise RuntimeError(f"No llama.cpp prebuilt asset matched this machine ({wanted})")


def _llama_cpp_asset_fragment() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"aarch64", "arm64"} else "x64"
    if system == "linux":
        return f"bin-ubuntu-{arch}.tar.gz"
    if system == "darwin":
        return f"bin-macos-{arch}.tar.gz"
    if system == "windows":
        return f"bin-win-cpu-{arch}.zip"
    raise RuntimeError(f"Unsupported platform for llama.cpp auto-install: {system}/{machine}")


def _llama_server_binary_name() -> str:
    return _binary_name("llama-server")


def _etcd_archive_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"aarch64", "arm64"} else "amd64"
    if system == "linux":
        return f"etcd-{ETCD_VERSION}-linux-{arch}.tar.gz"
    if system == "darwin":
        return f"etcd-{ETCD_VERSION}-darwin-{arch}.zip"
    if system == "windows":
        return f"etcd-{ETCD_VERSION}-windows-{arch}.zip"
    raise RuntimeError(f"Unsupported platform for etcd auto-install: {system}/{machine}")


def _etcd_download_url(archive_name: str) -> str:
    return f"https://github.com/etcd-io/etcd/releases/download/{ETCD_VERSION}/{archive_name}"


def _go_version_ok(go_bin: str) -> bool:
    result = subprocess.run(
        [go_bin, "version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        return False
    version = _parse_go_version(result.stdout)
    return version >= MIN_GO_VERSION


def _parse_go_version(raw: str) -> tuple[int, int]:
    marker = "go"
    for part in raw.split():
        if part.startswith(marker) and len(part) > len(marker):
            pieces = part.removeprefix(marker).split(".")
            if len(pieces) >= 2 and pieces[0].isdigit() and pieces[1].isdigit():
                return int(pieces[0]), int(pieces[1])
    return 0, 0


def _select_go_release() -> dict[str, str]:
    with urllib.request.urlopen(GO_RELEASE_API, timeout=30) as response:
        releases = json.loads(response.read().decode("utf-8"))
    if not isinstance(releases, list):
        raise RuntimeError("Could not read Go release metadata")

    wanted_os = _go_os()
    wanted_arch = _go_arch()
    for release in releases:
        if not isinstance(release, dict) or not release.get("stable"):
            continue
        version = str(release.get("version", ""))
        if _parse_go_version(version) < MIN_GO_VERSION:
            continue
        files = release.get("files")
        if not isinstance(files, list):
            continue
        for file_info in files:
            if not isinstance(file_info, dict):
                continue
            if file_info.get("os") != wanted_os or file_info.get("arch") != wanted_arch:
                continue
            kind = "zip" if wanted_os == "windows" else "archive"
            if file_info.get("kind") != kind:
                continue
            filename = str(file_info["filename"])
            return {
                "version": version,
                "filename": filename,
                "url": f"https://go.dev/dl/{filename}",
            }
    raise RuntimeError(f"No Go release matched this machine ({wanted_os}/{wanted_arch})")


def _go_os() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "darwin"
    if system == "linux":
        return "linux"
    if system == "windows":
        return "windows"
    raise RuntimeError(f"Unsupported platform for Go auto-install: {system}")


def _go_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "amd64"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    raise RuntimeError(f"Unsupported architecture for Go auto-install: {machine}")


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(destination)


def _extract_archive(archive_path: Path, install_dir: Path) -> None:
    install_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(install_dir)
        return
    with tarfile.open(archive_path) as archive:
        archive.extractall(install_dir)


def _find_llama_server(root: Path) -> Path | None:
    binary_name = _llama_server_binary_name()
    return _find_named_binary(root, binary_name)


def _find_named_binary(root: Path, binary_name: str) -> Path | None:
    matches = [path for path in root.rglob(binary_name) if path.is_file()]
    if not matches:
        return None
    matches.sort(key=lambda path: len(path.parts))
    return matches[0]


def doctor_checks(config: RelayConfig | None) -> list[CheckResult]:
    """Return environment checks for a loaded or missing config."""
    paths = RelayPaths.from_home()
    checks: list[CheckResult] = [
        CheckResult("relay home", paths.home.exists(), str(paths.home)),
        CheckResult("config", paths.config.exists(), str(paths.config)),
    ]
    if config is None:
        return checks

    if config.network.backend == "tailscale":
        tailscale = shutil.which("tailscale")
        checks.append(
            CheckResult(
                "tailscale",
                True,
                (
                    f"{tailscale} available"
                    if tailscale
                    else "not installed; only needed for cross-machine/cross-network clusters"
                ),
            )
        )
    else:
        checks.append(CheckResult("lan network", True, "configured"))

    if config.runs_coordinator:
        etcd = resolve_executable(config.membership.etcd_bin, "etcd")
        checks.append(
            CheckResult("etcd", True, etcd or "missing now; relay init/start will install")
        )
        go_bin = resolve_executable("go", "go")
        checks.append(
            CheckResult(
                "go",
                True,
                go_bin or "missing now; relay init/start will install managed Go",
            )
        )
        membership = resolve_executable(config.membership.service_bin, "membership-etcd")
        checks.append(
            CheckResult(
                "membership-etcd",
                True,
                membership or "missing now; relay init/start will build",
            )
        )

    if config.runs_worker:
        llama_server = resolve_executable(config.engine.server_bin, "llama-server")
        checks.append(
            CheckResult(
                "llama-server",
                True,
                llama_server or "missing now; relay start will auto-install llama.cpp",
            )
        )
        model = config.active_model()
        if model is None:
            checks.append(CheckResult("model", False, "no model configured"))
        else:
            model_path = Path(model.path).expanduser()
            checks.append(CheckResult("model", model_path.exists(), str(model_path)))

    return checks
