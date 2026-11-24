"""Run a recorded session through DSP and the pointer.

Usage (from the repository root, with the venv active):

    python scripts/pointer_replay.py data/raw/sessions/<name> [--engaged] [--plot-dir DIR]
    python scripts/pointer_replay.py data/raw/sessions/<name> --live [--dry-run]

Offline (default): processes the whole session as fast as possible, prints a summary, the
processing cost per frame and a digest of the cursor moves (same session -> same digest),
and saves a plot of the cursor path.

--live: plays the packets back with their original arrival timing through the same playout
buffer, pointer loop and dashboard as pointer_live.py, and moves the real cursor (unless
--dry-run). Useful to tune the pointer without the watch.

--engaged starts with the clutch engaged (recordings without double knocks would otherwise
never move the cursor).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from airhand.config import DspConfig, IngestConfig, PointerConfig  # noqa: E402
from airhand.dsp.processor import DspProcessor  # noqa: E402
from airhand.ingest.chunk import SampleChunk  # noqa: E402
from airhand.ingest.pipeline import build_pipeline  # noqa: E402
from airhand.ingest.playout import PlayoutBuffer  # noqa: E402
from airhand.ingest.recorder import read_session  # noqa: E402
from airhand.ingest.resampler import Resampler, resample_offline  # noqa: E402
from airhand.latency import StageTimer  # noqa: E402
from airhand.pointer.processor import PointerProcessor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path)
    parser.add_argument("--ingest-config", type=Path, default=CONFIGS / "ingest.yaml")
    parser.add_argument("--dsp-config", type=Path, default=CONFIGS / "dsp.yaml")
    parser.add_argument("--pointer-config", type=Path, default=CONFIGS / "pointer.yaml")
    parser.add_argument("--engaged", action="store_true", help="Start with the clutch engaged")
    parser.add_argument("--plot-dir", type=Path, default=REPO_ROOT / "data" / "interim" / "pointer")
    parser.add_argument("--live", action="store_true", help="Real-time replay with dashboard")
    parser.add_argument("--dry-run", action="store_true", help="With --live: no real cursor")
    return parser.parse_args()


def run_offline(
    chunks: list[SampleChunk],
    ingest: IngestConfig,
    dsp_config: DspConfig,
    pointer_config: PointerConfig,
    name: str,
    plot_dir: Path,
) -> None:
    resampler = Resampler(
        input_channels=ingest.record_channels,
        output_channels=ingest.resample_channels,
        quaternion_channels=ingest.quaternion_channels,
        grid_hz=ingest.grid_hz,
        max_gap_ms=ingest.max_interp_gap_ms,
    )
    frames = resample_offline(chunks, resampler)
    fs = ingest.grid_hz

    t0 = time.perf_counter_ns()
    dsp = DspProcessor(dsp_config, fs).process(frames)
    t1 = time.perf_counter_ns()
    out = PointerProcessor(pointer_config, fs).process(dsp)
    t2 = time.perf_counter_ns()

    n = len(frames)
    x = np.cumsum(out.dx)
    y = np.cumsum(out.dy)
    step = np.hypot(out.dx, out.dy)
    digest = hashlib.sha256(out.dx.tobytes() + out.dy.tobytes()).hexdigest()[:12]
    toggles = (out.time_ns[out.toggled] - out.time_ns[0]) / 1e9
    print(f"session          {name}")
    print(f"frames           {n} ({n / fs:.1f} s), invalid {int((~frames.valid).sum())}")
    print(f"engaged          {100 * out.engaged.mean():.1f} % of frames")
    print(f"double knocks    {len(toggles)} at {np.round(toggles, 2).tolist()} s")
    print(f"cursor travel    {step.sum():.0f} px, final offset ({x[-1]}, {y[-1]}) px")
    print(f"dsp cost         {(t1 - t0) / n / 1e3:.1f} us per frame (offline, Python loop)")
    print(f"pointer cost     {(t2 - t1) / n / 1e3:.1f} us per frame")
    print(f"move digest      {digest}")

    plot_dir.mkdir(parents=True, exist_ok=True)
    time_s = (out.time_ns - out.time_ns[0]) / 1e9
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=INK2)
    axes[0].plot(x, y, color=BLUE, linewidth=1)
    axes[0].invert_yaxis()  # screen y grows downwards
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_title("Cursor path (px, relative to start)", color=INK, loc="left")
    # Zoom on the 3 s around the fastest movement, where the filter's behaviour is visible.
    speed = np.nan_to_num(np.hypot(*out.velocity_px_s.T)) * out.engaged
    centre = time_s[int(np.argmax(speed))]
    window = (time_s > centre - 1.5) & (time_s < centre + 1.5) & out.engaged
    axes[1].plot(time_s[window], out.position_raw[window, 0], color=INK2, lw=0.8,
                 label="integrated")  # fmt: skip
    axes[1].plot(time_s[window], out.position_smooth[window, 0], color=ORANGE, lw=1.4,
                 label="One Euro")  # fmt: skip
    axes[1].set_title("Cursor x around the fastest move (px)", color=INK, loc="left")
    axes[1].set_xlabel("time (s)", color=INK2)
    axes[1].legend(frameon=True, facecolor=SURFACE, edgecolor=GRID)
    fig.tight_layout()
    path = plot_dir / f"pointer_{name}.png"
    fig.savefig(path, dpi=120)
    print(f"plot             {path}")


def run_live(
    chunks: list[SampleChunk],
    ingest: IngestConfig,
    dsp_config: DspConfig,
    pointer_config: PointerConfig,
    dry_run: bool,
) -> None:
    # Reuse the live script's wiring so the replay exercises exactly the live code path.
    sys.path.insert(0, str(Path(__file__).parent))
    from airhand.dispatch.windows_input import HotkeyPoller
    from airhand.pointer.engine import PointerEngine
    from airhand.ui.telemetry import TelemetryBuffer
    from pointer_live import make_mouse, run_with_dashboard

    fs = ingest.grid_hz
    timer = StageTimer()
    playout = PlayoutBuffer(
        ingest.resample_channels,
        round(1e9 / fs),
        pointer_config.playout_delay_s,
        pointer_config.playout_offset_window_s,
    )
    telemetry = TelemetryBuffer(round(pointer_config.telemetry_window_s * fs))
    hotkey = HotkeyPoller(pointer_config.clutch_hotkey) if sys.platform == "win32" else None
    engine = PointerEngine(
        playout=playout,
        dsp=DspProcessor(dsp_config, fs),
        pointer=PointerProcessor(pointer_config, fs),
        mouse=make_mouse(dry_run),
        timer=timer,
        telemetry=telemetry,
        hotkey_pressed=hotkey.pressed if hotkey is not None else None,
        tick_hz=pointer_config.tick_hz,
    )
    pipeline = build_pipeline(
        ingest, on_frames=lambda f: playout.push(f, time.monotonic_ns()), timer=timer
    )
    stop = threading.Event()

    def feed() -> None:
        """Hand each packet to the pipeline at its original arrival time."""
        start = time.monotonic_ns()
        first = chunks[0].recv_mono_ns
        for chunk in chunks:
            wait_ns = (chunk.recv_mono_ns - first) - (time.monotonic_ns() - start)
            if stop.wait(max(wait_ns, 0) / 1e9):
                return
            pipeline.process(chunk)

    feeder = threading.Thread(target=feed, daemon=True)
    engine.start()
    feeder.start()
    try:
        run_with_dashboard(telemetry, pointer_config)
    finally:
        stop.set()
        engine.stop()


def main() -> int:
    args = parse_args()
    ingest = IngestConfig.from_yaml(args.ingest_config)
    dsp_config = DspConfig.from_yaml(args.dsp_config)
    pointer_config = PointerConfig.from_yaml(args.pointer_config)
    if args.engaged:
        pointer_config = dataclasses.replace(pointer_config, start_engaged=True)
    session = read_session(args.session)
    if args.live:
        run_live(session.chunks, ingest, dsp_config, pointer_config, args.dry_run)
    else:
        run_offline(
            session.chunks, ingest, dsp_config, pointer_config, args.session.name, args.plot_dir
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
