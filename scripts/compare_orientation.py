"""Compare our DSP against the watch on recorded sessions (Phase 2 results).

Usage (from the repository root, with the venv active):

    python scripts/compare_orientation.py --session S1=data/raw/sessions/<a> \
        --session S2=data/raw/sessions/<b> --plot-dir docs/img/phase2

For each session it reports:
- Madgwick vs the watch's own orientation: tilt error and heading difference over time
  (both start from the watch's orientation, so this measures how far they drift apart),
- the gyro bias estimate and how much of the session counted as still,
- the knock signal of everyday motion vs the knock threshold, and any double knocks the
  detector found (in recordings without deliberate knocks, each one is a false trigger).
Then a Madgwick gain (beta) sweep on the last session. Prints markdown tables.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from airhand.config import DspConfig, IngestConfig, PointerConfig  # noqa: E402
from airhand.dsp.processor import DspOutput, DspProcessor  # noqa: E402
from airhand.ingest.chunk import Frames  # noqa: E402
from airhand.ingest.recorder import read_session  # noqa: E402
from airhand.ingest.resampler import Resampler, resample_offline  # noqa: E402
from airhand.pointer.processor import PointerProcessor  # noqa: E402
from airhand.ui.telemetry import orientation_errors  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BETAS = (0.0, 0.01, 0.04, 0.1, 0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", action="append", required=True, help="LABEL=path")
    parser.add_argument("--plot-dir", type=Path, default=REPO_ROOT / "data" / "interim" / "phase2")
    return parser.parse_args()


def load_frames(path: Path, ingest: IngestConfig) -> Frames:
    resampler = Resampler(
        input_channels=ingest.record_channels,
        output_channels=ingest.resample_channels,
        quaternion_channels=ingest.quaternion_channels,
        grid_hz=ingest.grid_hz,
        max_gap_ms=ingest.max_interp_gap_ms,
    )
    return resample_offline(read_session(path).chunks, resampler)


def drift_deg_per_min(time_s: np.ndarray, heading: np.ndarray) -> float:
    ok = np.isfinite(heading)
    slope = np.polyfit(time_s[ok], np.unwrap(np.radians(heading[ok])), 1)[0]
    return float(np.degrees(slope) * 60.0)


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=INK2)


def main() -> int:
    args = parse_args()
    ingest = IngestConfig.from_yaml(CONFIGS / "ingest.yaml")
    dsp_config = DspConfig.from_yaml(CONFIGS / "dsp.yaml")
    pointer_config = PointerConfig.from_yaml(CONFIGS / "pointer.yaml")
    fs = ingest.grid_hz
    sessions = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.session]

    orientation_rows, signal_rows = [], []
    curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    frames_by_label: dict[str, Frames] = {}
    for label, path in sessions:
        frames = load_frames(path, ingest)
        frames_by_label[label] = frames
        dsp: DspOutput = DspProcessor(dsp_config, fs).process(frames)
        tilt, heading = orientation_errors(dsp)
        time_s = (dsp.time_ns - dsp.time_ns[0]) / 1e9
        curves.append((label, time_s, tilt, heading))
        bias_dps = np.degrees(np.nanmax(np.abs(dsp.gyro_bias), axis=0))
        orientation_rows.append(
            f"| {label} | {time_s[-1]:.0f} | {np.nanpercentile(tilt, 50):.2f} | "
            f"{np.nanpercentile(tilt, 95):.2f} | {np.nanmax(tilt):.2f} | "
            f"{drift_deg_per_min(time_s, heading):+.2f} | {np.nanmax(np.abs(heading)):.2f} | "
            f"{100 * dsp.still.mean():.0f} | {bias_dps.max():.3f} |"
        )
        pointer = PointerProcessor(pointer_config, fs).process(dsp)
        knock = pointer.knock_signal_g
        signal_rows.append(
            f"| {label} | {np.nanpercentile(knock, 99):.2f} | {np.nanpercentile(knock, 99.9):.2f}"
            f" | {np.nanmax(knock):.2f} | {int(pointer.toggled.sum())} |"
        )

    print("| Session | Length (s) | Tilt err p50 (deg) | p95 | max | Heading drift (deg/min) "
          "| Max heading diff (deg) | Still (%) | Max bias estimate (deg/s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(orientation_rows))
    print(f"\nKnock threshold {pointer_config.knock_threshold_g} g")
    print("| Session | Knock signal p99 (g) | p99.9 | max | Double knocks found |")
    print("|---|---|---|---|---|")
    print("\n".join(signal_rows))

    label, frames = list(frames_by_label.items())[-1]
    print(f"\nMadgwick beta sweep on {label}")
    print("| beta | Tilt err p50 (deg) | p95 | Heading drift (deg/min) |")
    print("|---|---|---|---|")
    for beta in BETAS:
        config = dataclasses.replace(dsp_config, madgwick_beta=beta)
        dsp = DspProcessor(config, fs).process(frames)
        tilt, heading = orientation_errors(dsp)
        time_s = (dsp.time_ns - dsp.time_ns[0]) / 1e9
        print(
            f"| {beta} | {np.nanpercentile(tilt, 50):.2f} | {np.nanpercentile(tilt, 95):.2f} | "
            f"{drift_deg_per_min(time_s, heading):+.2f} |"
        )

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(curves), 1, figsize=(11, 2.6 * len(curves)), sharex=False,
                             facecolor=SURFACE, squeeze=False)  # fmt: skip
    for ax, (label, time_s, tilt, heading) in zip(axes[:, 0], curves, strict=True):
        style(ax)
        ax.plot(time_s, tilt, color=BLUE, lw=0.9, label="tilt error")
        ax.plot(time_s, heading, color=ORANGE, lw=0.9, label="heading difference")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{label}: Madgwick vs watch orientation (deg)", color=INK, loc="left")
        ax.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor=GRID)
    axes[-1, 0].set_xlabel("time (s)", color=INK2)
    fig.tight_layout()
    path = args.plot_dir / "orientation_vs_watch.png"
    fig.savefig(path, dpi=120)
    print(f"\nplot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
