"""Typed loading of YAML configuration files.

Configs are plain YAML so they can be edited without touching code. Each consumer gets a
small frozen dataclass so typos in key names fail loudly at load time instead of turning
into silent defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file that must contain a mapping at the top level."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


@dataclass(frozen=True)
class ProbeConfig:
    """Settings for the Phase 0 probe (see configs/probe.yaml for explanations)."""

    bind_host: str
    http_port: int
    udp_port: int
    output_root: str
    preview_chars: int
    print_each_packet: bool
    stats_interval_s: float
    udp_max_datagram_bytes: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbeConfig:
        expected = {f.name for f in fields(cls)}
        missing = expected - data.keys()
        unknown = data.keys() - expected
        if missing or unknown:
            raise ValueError(
                f"Bad probe config. Missing keys: {sorted(missing)}; "
                f"unknown keys: {sorted(unknown)}"
            )
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path) -> ProbeConfig:
        return cls.from_dict(load_yaml(path))
