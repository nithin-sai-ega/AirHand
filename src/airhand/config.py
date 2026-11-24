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


def check_keys(config_name: str, expected: set[str], data: dict[str, Any]) -> None:
    """Fail loudly on missing or unknown (usually misspelled) config keys."""
    missing = expected - data.keys()
    unknown = data.keys() - expected
    if missing or unknown:
        raise ValueError(
            f"Bad {config_name} config. Missing keys: {sorted(missing)}; "
            f"unknown keys: {sorted(unknown)}"
        )


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
        check_keys("probe", {f.name for f in fields(cls)}, data)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path) -> ProbeConfig:
        return cls.from_dict(load_yaml(path))


@dataclass(frozen=True)
class Phase0AnalysisConfig:
    """Settings for the Phase 0 stream analysis (see configs/phase0_analysis.yaml)."""

    probe_root: str
    sensor: str
    report_sessions: dict[str, str]
    still_session: str
    still_window_s: list[float]
    rotation_fields: list[str]
    comparison_session: str
    comparison_phone_sensor: str
    gap_factor: float
    envelope_window_s: float
    envelope_settle_s: float
    percentiles: list[float]
    plot_dir: str
    summary_path: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Phase0AnalysisConfig:
        check_keys("phase 0 analysis", {f.name for f in fields(cls)}, data)
        config = cls(**data)
        if config.still_session not in config.report_sessions:
            raise ValueError("still_session must be one of the report_sessions labels")
        if len(config.still_window_s) != 2 or config.still_window_s[0] >= config.still_window_s[1]:
            raise ValueError("still_window_s must be [start_s, end_s] with start < end")
        return config

    @classmethod
    def from_yaml(cls, path: Path) -> Phase0AnalysisConfig:
        return cls.from_dict(load_yaml(path))


@dataclass(frozen=True)
class IngestConfig:
    """Settings for the Phase 1 ingest pipeline (see configs/ingest.yaml)."""

    bind_host: str
    http_port: int
    data_path: str
    sensor: str
    source_app: str
    record_channels: tuple[str, ...]
    resample_channels: tuple[str, ...]
    quaternion_channels: tuple[str, ...]
    grid_hz: float
    max_interp_gap_ms: float
    ring_capacity_s: float
    queue_max_chunks: int
    output_root: str
    flush_interval_s: float
    stats_interval_s: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestConfig:
        check_keys("ingest", {f.name for f in fields(cls)}, data)
        values = dict(data)
        # YAML lists become tuples so channel orders compare equal to chunk channels.
        for key in ("record_channels", "resample_channels", "quaternion_channels"):
            values[key] = tuple(values[key])
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: Path) -> IngestConfig:
        return cls.from_dict(load_yaml(path))

    @property
    def ring_capacity_frames(self) -> int:
        return round(self.ring_capacity_s * self.grid_hz)


@dataclass(frozen=True)
class DspConfig:
    """Settings for the Phase 2 signal processing stage (see configs/dsp.yaml)."""

    gyro_channels: tuple[str, ...]
    user_accel_channels: tuple[str, ...]
    gravity_channels: tuple[str, ...]
    quaternion_channels: tuple[str, ...]
    lowpass_cutoff_hz: float
    lowpass_order: int
    stillness_window_s: float
    stillness_gyro_dps: float
    stillness_accel_g: float
    bias_time_constant_s: float
    madgwick_beta: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DspConfig:
        check_keys("dsp", {f.name for f in fields(cls)}, data)
        values = dict(data)
        for key in ("gyro_channels", "user_accel_channels", "gravity_channels"):
            values[key] = tuple(values[key])
            if len(values[key]) != 3:
                raise ValueError(f"{key} must name exactly 3 channels (x, y, z)")
        values["quaternion_channels"] = tuple(values["quaternion_channels"])
        if len(values["quaternion_channels"]) != 4:
            raise ValueError("quaternion_channels must name exactly 4 channels (w, x, y, z)")
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: Path) -> DspConfig:
        return cls.from_dict(load_yaml(path))


@dataclass(frozen=True)
class PointerConfig:
    """Settings for the Phase 2 pointer, clutch and live loop (see configs/pointer.yaml)."""

    forearm_axis: tuple[float, float, float]
    min_horizontal_norm: float
    dead_zone_dps: float
    gain_low_px_per_deg: float
    gain_high_px_per_deg: float
    gain_mid_dps: float
    gain_width_dps: float
    one_euro_min_cutoff_hz: float
    one_euro_beta: float
    one_euro_d_cutoff_hz: float
    clutch_hotkey: str
    start_engaged: bool
    knock_highpass_hz: float
    knock_threshold_g: float
    knock_min_spacing_s: float
    knock_pair_min_s: float
    knock_pair_max_s: float
    knock_quiet_before_s: float
    knock_quiet_after_s: float
    knock_cooldown_s: float
    playout_delay_s: float
    playout_offset_window_s: float
    tick_hz: float
    telemetry_window_s: float
    dashboard_refresh_hz: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PointerConfig:
        check_keys("pointer", {f.name for f in fields(cls)}, data)
        values = dict(data)
        axis = tuple(float(v) for v in values["forearm_axis"])
        if len(axis) != 3 or not any(axis):
            raise ValueError("forearm_axis must be a non-zero [x, y, z] vector")
        values["forearm_axis"] = axis
        if values["knock_pair_min_s"] >= values["knock_pair_max_s"]:
            raise ValueError("knock_pair_min_s must be smaller than knock_pair_max_s")
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: Path) -> PointerConfig:
        return cls.from_dict(load_yaml(path))
