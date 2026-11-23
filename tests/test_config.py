"""Tests for typed config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from airhand.config import ProbeConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repository_probe_config_loads() -> None:
    config = ProbeConfig.from_yaml(REPO_ROOT / "configs" / "probe.yaml")
    assert 0 < config.http_port < 65536
    assert 0 < config.udp_port < 65536
    assert config.stats_interval_s > 0


def test_missing_and_unknown_keys_are_reported() -> None:
    good = ProbeConfig.from_yaml(REPO_ROOT / "configs" / "probe.yaml")
    data = {k: v for k, v in good.__dict__.items() if k != "http_port"}
    data["htp_port"] = 8000  # typo
    with pytest.raises(ValueError, match="http_port.*htp_port"):
        ProbeConfig.from_dict(data)
