"""Phase 0 stream analysis: measure the real Sensor Logger / Apple Watch stream.

Usage (from the repository root, with the venv active):

    python scripts/phase0_analyze.py

Reads the sessions listed in configs/phase0_analysis.yaml, prints markdown tables (pasted
into docs/RESULTS.md), writes plots to docs/img/phase0/ and a JSON summary to data/interim/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # render to files only; no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from airhand.config import Phase0AnalysisConfig  # noqa: E402
from airhand.ingest.probe_session import ProbeSession, load_probe_session  # noqa: E402
from airhand.ingest.stream_stats import (  # noqa: E402
    NS_PER_S,
    count_missing_message_ids,
    drift_ppm,
    fit_line,
    gap_stats,
    inter_arrival_ms,
    interval_stats,
    lower_envelope,
    sample_age_ms,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase0_analysis.yaml"

# Chart colours: light chart surface, ink, and the first three categorical slots.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
RAD_TO_DEG = 180.0 / np.pi
# Y range of the still-noise plot; a presentation choice, not an analysis parameter.
STILL_PLOT_LIMIT_DEG_S = 2.0


# ---------------------------------------------------------------- per-packet ages


def packet_ages(session: ProbeSession, sensor: str) -> pd.DataFrame:
    """Per packet: arrival time and the age of its freshest and oldest sample of `sensor`.

    The freshest sample's age is the delay the batching cannot explain (transport plus
    clock offset); the oldest sample's age adds the time spent waiting for the batch.
    """
    samples = session.sensor(sensor)[["packet_seq", "time_ns"]]
    per_packet = samples.groupby("packet_seq")["time_ns"].agg(newest="max", oldest="min")
    merged = session.packets.set_index("packet_seq").join(per_packet, how="inner")
    recv = merged["recv_wall_ns"].to_numpy()
    return pd.DataFrame(
        {
            "t_s": (merged["recv_mono_ns"] - merged["recv_mono_ns"].iloc[0]).to_numpy() / NS_PER_S,
            "freshest_age_ms": sample_age_ms(merged["newest"].to_numpy(), recv),
            "oldest_age_ms": sample_age_ms(merged["oldest"].to_numpy(), recv),
        }
    )


# ---------------------------------------------------------------- per-session summary


def summarize_session(
    label: str, session: ProbeSession, config: Phase0AnalysisConfig
) -> dict[str, Any]:
    stream = session.sensor(config.sensor)
    time_ns = stream["time_ns"].to_numpy()
    # Arrival order (packet order, then position in packet) is needed to spot reordering.
    arrival_order = session.samples[session.samples["sensor"] == config.sensor]["time_ns"]

    intervals = interval_stats(arrival_order.to_numpy())
    gaps = gap_stats(time_ns, config.gap_factor)
    arrivals = inter_arrival_ms(session.packets["recv_mono_ns"].to_numpy())
    per_packet_counts = stream.groupby("packet_seq").size().to_numpy()
    all_ages = sample_age_ms(
        time_ns,
        stream["packet_seq"]
        .map(session.packets.set_index("packet_seq")["recv_wall_ns"])
        .to_numpy(),
    )
    ages = packet_ages(session, config.sensor)
    q = config.percentiles

    return {
        "label": label,
        "session": session.name,
        "note": session.note,
        "intervals": asdict(intervals),
        "gaps": asdict(gaps),
        "packets": {
            "n_packets": len(session.packets),
            "missing_message_ids": count_missing_message_ids(
                session.packets["message_id"].to_numpy()
            ),
            "inter_arrival_ms_percentiles": dict(zip(q, np.percentile(arrivals, q), strict=True)),
            "inter_arrival_ms_max": float(arrivals.max()),
            "samples_per_packet_percentiles": dict(
                zip(q, np.percentile(per_packet_counts, q), strict=True)
            ),
            "samples_per_packet_max": int(per_packet_counts.max()),
            "body_kib_median": float(session.packets["body_len"].median() / 1024),
        },
        "age_ms": {
            "all_samples_percentiles": dict(zip(q, np.percentile(all_ages, q), strict=True)),
            "freshest_min": float(ages["freshest_age_ms"].min()),
            "freshest_median": float(ages["freshest_age_ms"].median()),
            "freshest_max": float(ages["freshest_age_ms"].max()),
        },
    }


def seconds_from_start(stream: pd.DataFrame) -> np.ndarray:
    return (stream["time_ns"] - stream["time_ns"].iloc[0]).to_numpy() / NS_PER_S


def still_noise(session: ProbeSession, config: Phase0AnalysisConfig) -> dict[str, dict[str, float]]:
    """Mean (bias) and standard deviation (noise) of rotation rate while the watch is still.

    Only the configured undisturbed window is used, so a bump of the desk does not
    inflate the noise estimate.
    """
    stream = session.sensor(config.sensor)
    t_s = seconds_from_start(stream)
    start_s, end_s = config.still_window_s
    quiet = stream[(t_s >= start_s) & (t_s <= end_s)]
    result = {}
    for field in config.rotation_fields:
        values = quiet[field].to_numpy()
        result[field] = {
            "mean_deg_s": float(values.mean() * RAD_TO_DEG),
            "std_deg_s": float(values.std() * RAD_TO_DEG),
            "n_samples": len(values),
        }
    return result


def envelope_fit(ages: pd.DataFrame, window_s: float, settle_s: float) -> dict[str, float]:
    """Line fit through the lower envelope of the freshest-sample age, after the start-up
    backlog. A slope here combines clock drift with any slow change in transport delay.
    """
    steady = ages[ages["t_s"] >= settle_s]
    centers, minima = lower_envelope(
        steady["t_s"].to_numpy(), steady["freshest_age_ms"].to_numpy(), window_s
    )
    fit = fit_line(centers, minima)
    return {
        "settle_s": settle_s,
        "envelope_intercept_ms": fit.intercept,
        "slope_ppm": drift_ppm(fit),
        "fit_start_s": float(steady["t_s"].min()),
        "fit_end_s": float(steady["t_s"].max()),
    }


def phone_vs_watch(session: ProbeSession, config: Phase0AnalysisConfig) -> dict[str, Any]:
    phone = packet_ages(session, config.comparison_phone_sensor)
    watch = packet_ages(session, config.sensor)
    return {
        "session": session.name,
        "n_packets": len(watch),
        "phone_freshest_median_ms": float(phone["freshest_age_ms"].median()),
        "watch_freshest_median_ms": float(watch["freshest_age_ms"].median()),
        "watch_minus_phone_median_ms": float(
            watch["freshest_age_ms"].median() - phone["freshest_age_ms"].median()
        ),
    }


# ---------------------------------------------------------------- plots


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def add_legend(ax: plt.Axes, ncols: int) -> None:
    """Legend on an opaque surface-coloured box so gridlines don't run through the text."""
    ax.legend(
        fontsize=8,
        labelcolor=INK_SECONDARY,
        ncols=ncols,
        loc="upper right",
        frameon=True,
        facecolor=SURFACE,
        edgecolor="none",
        framealpha=1.0,
    )


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def plot_sample_intervals(sessions: dict[str, ProbeSession], sensor: str, path: Path) -> None:
    dt_ms = np.concatenate(
        [np.diff(np.sort(s.sensor(sensor)["time_ns"].to_numpy())) / 1e6 for s in sessions.values()]
    )
    fig, ax = new_figure()
    ax.hist(dt_ms, bins=np.arange(0, 40.5, 0.5), color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    style_axes(
        ax,
        f"Watch sample spacing, {len(dt_ms):,} intervals (S1-S3): all at {np.median(dt_ms):.2f} ms",
        "time between consecutive samples (ms)",
        "count",
    )
    save(fig, path)


def plot_packet_intervals(sessions: dict[str, ProbeSession], path: Path) -> None:
    arrivals = np.concatenate(
        [inter_arrival_ms(s.packets["recv_mono_ns"].to_numpy()) for s in sessions.values()]
    )
    fig, ax = new_figure()
    ax.hist(arrivals, bins=np.arange(0, 2600, 50), color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    style_axes(
        ax,
        f"Time between packets arriving at the laptop ({len(arrivals)} gaps, S1-S3)",
        "packet inter-arrival time (ms)",
        "count",
    )
    save(fig, path)


def plot_samples_per_packet(sessions: dict[str, ProbeSession], sensor: str, path: Path) -> None:
    counts = np.concatenate(
        [s.sensor(sensor).groupby("packet_seq").size().to_numpy() for s in sessions.values()]
    )
    values, freq = np.unique(counts, return_counts=True)
    fig, ax = new_figure()
    ax.bar(values, freq, width=6, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    for v, f in zip(values, freq, strict=True):
        ax.annotate(
            f"{v}",
            (v, f),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    style_axes(ax, "Watch samples per packet (S1-S3)", "samples in one packet", "packets")
    save(fig, path)


def plot_sample_age(ages: pd.DataFrame, fit: dict[str, float], label: str, path: Path) -> None:
    fig, ax = new_figure()
    ax.axvspan(0, fit["settle_s"], color=GRID, alpha=0.6, linewidth=0, label="start-up backlog")
    ax.scatter(
        ages["t_s"],
        ages["oldest_age_ms"],
        s=14,
        color=SERIES[1],
        edgecolors=SURFACE,
        linewidths=0.8,
        label="oldest sample in packet",
        zorder=3,
    )
    ax.scatter(
        ages["t_s"],
        ages["freshest_age_ms"],
        s=14,
        color=SERIES[0],
        edgecolors=SURFACE,
        linewidths=0.8,
        label="freshest sample in packet",
        zorder=3,
    )
    t = np.array([fit["fit_start_s"], fit["fit_end_s"]])
    line = fit["envelope_intercept_ms"] + fit["slope_ppm"] / 1000.0 * t
    ax.plot(t, line, color=SERIES[2], linewidth=2, label="lower-envelope fit (freshest)")
    add_legend(ax, ncols=1)
    style_axes(
        ax,
        f"Sample age when its packet reaches the laptop ({label})",
        "time since first packet (s)",
        "age (ms)",
    )
    save(fig, path)


def plot_phone_vs_watch(session: ProbeSession, config: Phase0AnalysisConfig, path: Path) -> None:
    phone = packet_ages(session, config.comparison_phone_sensor)["freshest_age_ms"]
    watch = packet_ages(session, config.sensor)["freshest_age_ms"]
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    rows = [
        ("iPhone gyroscope", phone, SERIES[1]),
        ("Apple Watch (wrist motion)", watch, SERIES[0]),
    ]
    for y, (_name, values, color) in enumerate(rows):
        ax.scatter(
            values,
            np.full(len(values), y),
            s=30,
            color=color,
            edgecolors=SURFACE,
            linewidths=1,
            zorder=3,
        )
        ax.annotate(
            f"median {values.median():.0f} ms",
            (values.median(), y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_yticks(range(len(rows)), [name for name, _, _ in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    style_axes(ax, "Freshest-sample age per packet, same session and clocks", "age (ms)", "")
    save(fig, path)


def plot_still_noise(session: ProbeSession, config: Phase0AnalysisConfig, path: Path) -> None:
    """Full still session, zoomed to the noise level; the shaded span is what was measured.

    The y-axis is clipped on purpose: a bump of the watch reaches tens of deg/s and would
    flatten the noise into an invisible line.
    """
    stream = session.sensor(config.sensor)
    t_s = seconds_from_start(stream)
    start_s, end_s = config.still_window_s
    fig, ax = new_figure()
    ax.axvspan(start_s, end_s, color=GRID, alpha=0.6, linewidth=0, label="window used for noise")
    for field, color in zip(config.rotation_fields, SERIES, strict=False):
        ax.plot(t_s, stream[field].to_numpy() * RAD_TO_DEG, color=color, linewidth=1, label=field)
    ax.set_ylim(-STILL_PLOT_LIMIT_DEG_S, STILL_PLOT_LIMIT_DEG_S)
    add_legend(ax, ncols=4)
    style_axes(
        ax,
        f"Rotation rate, watch still on the desk ({config.still_session}); "
        f"y clipped at +/-{STILL_PLOT_LIMIT_DEG_S:g} deg/s",
        "time (s)",
        "rotation rate (deg/s)",
    )
    save(fig, path)


# ---------------------------------------------------------------- report


def markdown_report(
    summaries: list[dict[str, Any]],
    noise: dict[str, Any],
    fit: dict[str, float],
    fit_label: str,
    comparison: dict[str, Any],
) -> str:
    lines = [
        "| | " + " | ".join(s["label"] for s in summaries) + " |",
        "|---|" + "---|" * len(summaries),
    ]

    def row(name: str, fmt: str, getter: Any) -> None:
        lines.append(f"| {name} | " + " | ".join(fmt.format(getter(s)) for s in summaries) + " |")

    row("Duration (s)", "{:.1f}", lambda s: s["intervals"]["span_s"])
    row("Watch samples", "{:,}", lambda s: s["intervals"]["n_samples"])
    row("Rate from median spacing (Hz)", "{:.2f}", lambda s: s["intervals"]["rate_from_median_hz"])
    row("Rate from count / duration (Hz)", "{:.2f}", lambda s: s["intervals"]["rate_from_count_hz"])
    row("Spacing std (ms)", "{:.3f}", lambda s: s["intervals"]["std_dt_ms"])
    row("Out-of-order or duplicate samples", "{}", lambda s: s["intervals"]["n_non_increasing"])
    row("Gaps (> 1.5 x period)", "{}", lambda s: s["gaps"]["n_gaps"])
    row("Lost messages (messageId gaps)", "{}", lambda s: s["packets"]["missing_message_ids"])
    row("Packets", "{}", lambda s: s["packets"]["n_packets"])
    row(
        "Packet interval p50 / p95 / max (ms)",
        "{}",
        lambda s: "{:.0f} / {:.0f} / {:.0f}".format(
            s["packets"]["inter_arrival_ms_percentiles"][50],
            s["packets"]["inter_arrival_ms_percentiles"][95],
            s["packets"]["inter_arrival_ms_max"],
        ),
    )
    row(
        "Samples per packet p50 / p95 / max",
        "{}",
        lambda s: "{:.0f} / {:.0f} / {}".format(
            s["packets"]["samples_per_packet_percentiles"][50],
            s["packets"]["samples_per_packet_percentiles"][95],
            s["packets"]["samples_per_packet_max"],
        ),
    )
    row("Packet size, median (KiB)", "{:.1f}", lambda s: s["packets"]["body_kib_median"])
    row(
        "Sample age p50 / p95 (ms)",
        "{}",
        lambda s: "{:.0f} / {:.0f}".format(
            s["age_ms"]["all_samples_percentiles"][50], s["age_ms"]["all_samples_percentiles"][95]
        ),
    )
    row(
        "Freshest-sample age min / p50 / max (ms)",
        "{}",
        lambda s: "{:.0f} / {:.0f} / {:.0f}".format(
            s["age_ms"]["freshest_min"], s["age_ms"]["freshest_median"], s["age_ms"]["freshest_max"]
        ),
    )

    lines += [
        "",
        "Still-watch rotation rate (bias = mean, noise = std):",
        "",
        "| Axis | Mean (deg/s) | Std (deg/s) |",
        "|---|---|---|",
    ]
    for field, values in noise.items():
        lines.append(f"| {field} | {values['mean_deg_s']:+.3f} | {values['std_deg_s']:.3f} |")

    lines += [
        "",
        f"Lower-envelope fit of freshest-sample age ({fit_label}, after {fit['settle_s']:g} s): "
        f"intercept {fit['envelope_intercept_ms']:.0f} ms, slope {fit['slope_ppm']:+.0f} ppm",
        "",
        f"Phone vs watch ({comparison['session']}, {comparison['n_packets']} packets): "
        f"freshest-sample age median phone {comparison['phone_freshest_median_ms']:.0f} ms, "
        f"watch {comparison['watch_freshest_median_ms']:.0f} ms, "
        f"difference {comparison['watch_minus_phone_median_ms']:.0f} ms",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Phase0AnalysisConfig.from_yaml(args.config)
    probe_root = REPO_ROOT / config.probe_root
    plot_dir = REPO_ROOT / config.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    sessions = {
        label: load_probe_session(probe_root / folder)
        for label, folder in config.report_sessions.items()
    }
    comparison_session = load_probe_session(probe_root / config.comparison_session)

    summaries = [summarize_session(label, s, config) for label, s in sessions.items()]
    noise = still_noise(sessions[config.still_session], config)
    # The longest session gives the most reliable envelope fit.
    longest_label = max(summaries, key=lambda s: s["intervals"]["span_s"])["label"]
    longest_ages = packet_ages(sessions[longest_label], config.sensor)
    fit = envelope_fit(longest_ages, config.envelope_window_s, config.envelope_settle_s)
    comparison = phone_vs_watch(comparison_session, config)

    plot_sample_intervals(sessions, config.sensor, plot_dir / "sample_intervals.png")
    plot_packet_intervals(sessions, plot_dir / "packet_intervals.png")
    plot_samples_per_packet(sessions, config.sensor, plot_dir / "samples_per_packet.png")
    plot_sample_age(longest_ages, fit, longest_label, plot_dir / "sample_age.png")
    plot_phone_vs_watch(comparison_session, config, plot_dir / "phone_vs_watch_age.png")
    plot_still_noise(sessions[config.still_session], config, plot_dir / "still_noise.png")

    summary_path = REPO_ROOT / config.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "sessions": summaries,
                "still_noise": noise,
                "envelope_fit": fit,
                "envelope_fit_session": longest_label,
                "phone_vs_watch": comparison,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {summary_path.relative_to(REPO_ROOT)}\n")
    print(markdown_report(summaries, noise, fit, longest_label, comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
