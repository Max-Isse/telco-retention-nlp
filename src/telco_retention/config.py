"""Load the YAML config and resolve paths relative to the project root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("TELCO_ROOT") or Path(__file__).resolve().parents[2])
DEFAULT_CONFIG = Path(os.environ.get("TELCO_CONFIG") or PROJECT_ROOT / "configs" / "config.yaml")


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.root / p

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw[name])


def load_config(path: Path | str | None = None, root: Path | None = None) -> Config:
    with Path(path or DEFAULT_CONFIG).open(encoding="utf-8") as f:
        return Config(raw=yaml.safe_load(f), root=root or PROJECT_ROOT)


def with_root(cfg: Config, root: Path) -> Config:
    """Same config, data and report paths under another root (tests)."""
    return Config(raw=cfg.raw, root=root)
