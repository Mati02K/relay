"""Filesystem paths used by the Relay CLI runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RELAY_HOME_ENV = "RELAY_HOME"
DEFAULT_RELAY_HOME = "~/.relay"


def relay_home() -> Path:
    """Return the Relay state directory, defaulting to ``~/.relay``."""
    return Path(os.getenv(RELAY_HOME_ENV, DEFAULT_RELAY_HOME)).expanduser()


@dataclass(frozen=True)
class RelayPaths:
    """Concrete runtime paths under a Relay state directory."""

    home: Path
    config: Path
    identity: Path
    trust_db: Path
    models: Path
    bin: Path
    logs: Path
    run: Path
    cache: Path
    etcd_data: Path

    @classmethod
    def from_home(cls, home: Path | None = None) -> RelayPaths:
        """Build path values for ``home`` or the configured default."""
        root = home or relay_home()
        return cls(
            home=root,
            config=root / "config.json",
            identity=root / "identity.json",
            trust_db=root / "trust.db",
            models=root / "models",
            bin=root / "bin",
            logs=root / "logs",
            run=root / "run",
            cache=root / "cache",
            etcd_data=root / "etcd-data",
        )

    def ensure(self) -> None:
        """Create directories needed by the CLI runtime."""
        for path in (
            self.home,
            self.models,
            self.bin,
            self.logs,
            self.run,
            self.cache,
            self.etcd_data,
        ):
            path.mkdir(parents=True, exist_ok=True)


def config_path() -> Path:
    """Return the current Relay config path."""
    return RelayPaths.from_home().config
