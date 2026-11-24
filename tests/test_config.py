"""Tests for typed config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from airhand.config import DspConfig, PointerConfig, ProbeConfig

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


def test_repository_dsp_and_pointer_configs_load() -> None:
    dsp = DspConfig.from_yaml(REPO_ROOT / "configs" / "dsp.yaml")
    pointer = PointerConfig.from_yaml(REPO_ROOT / "configs" / "pointer.yaml")
    assert len(dsp.gyro_channels) == 3
    assert len(dsp.quaternion_channels) == 4
    assert len(pointer.forearm_axis) == 3
    assert pointer.knock_pair_min_s < pointer.knock_pair_max_s


def test_pointer_config_rejects_bad_knock_window() -> None:
    good = PointerConfig.from_yaml(REPO_ROOT / "configs" / "pointer.yaml")
    data = dict(good.__dict__)
    data["knock_pair_min_s"] = data["knock_pair_max_s"]
    with pytest.raises(ValueError, match="knock_pair_min_s"):
        PointerConfig.from_dict(data)
