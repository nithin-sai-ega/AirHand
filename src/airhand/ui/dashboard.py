"""Live dashboard: what every pipeline stage is doing, updated a few dozen times per second.

Qt is the toolkit that draws the window; pyqtgraph draws fast-updating plots inside it.
The window only reads a TelemetryBuffer snapshot on a timer, so it never slows down the
pointer loop, which runs in its own thread.

Plots (shared time axis, newest frame at 0 s):
    1. raw gyro                       (what the watch sent)
    2. gyro after bias removal + low-pass
    3. knock signal with the threshold, and double-knock toggles
    4. cursor position: integrated (raw) vs One Euro smoothed
    5. Madgwick vs the watch's own orientation (tilt error, heading difference)
Side panel: clutch state, playout buffer, and per-stage latency percentiles.
"""

from __future__ import annotations

import signal

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from airhand.ui.telemetry import TelemetryBuffer, TelemetrySnapshot

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")

LATENCY_ROWS = (
    ("sample_age", "network (sample age)"),
    ("ingest", "ingest + resample / packet"),
    ("dsp", "dsp / tick"),
    ("pointer", "pointer / tick"),
    ("dispatch", "cursor move / tick"),
)


def _pen(color: str, width: float = 1.5, dashed: bool = False) -> QtGui.QPen:
    pen = pg.mkPen(color=color, width=width)
    if dashed:
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
    return pen


class Dashboard(QtWidgets.QMainWindow):
    def __init__(
        self,
        telemetry: TelemetryBuffer,
        *,
        refresh_hz: float,
        window_s: float,
        knock_threshold_g: float,
        playout_delay_s: float,
    ) -> None:
        super().__init__()
        self.telemetry = telemetry
        self.playout_delay_s = playout_delay_s
        self.setWindowTitle("AirHand pointer")
        self.resize(1300, 900)

        pg.setConfigOptions(antialias=True, background=SURFACE, foreground=INK_SECONDARY)
        plots = pg.GraphicsLayoutWidget()
        self.curves: dict[str, list[pg.PlotDataItem]] = {}

        def add_plot(row: int, title: str, unit: str) -> pg.PlotItem:
            plot = plots.addPlot(row=row, col=0, title=f"<span style='color:{INK}'>{title}</span>")
            plot.setLabel("left", unit)
            plot.getAxis("left").enableAutoSIPrefix(False)  # show 0.8 g, not 800 x0.001 g
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setXRange(-window_s, 0, padding=0)
            plot.addLegend(offset=(5, 5), brush=pg.mkBrush(SURFACE), pen=_pen(GRID))
            if row > 0:
                plot.setXLink(self.first_plot)
            return plot

        raw = add_plot(0, "Raw gyro", "deg/s")
        self.first_plot = raw
        clean = add_plot(1, "Gyro after bias removal and low-pass", "deg/s")
        knock = add_plot(2, "Knock signal (high-passed acceleration)", "g")
        cursor = add_plot(3, "Cursor position: integrated vs One Euro", "px")
        orient = add_plot(4, "Madgwick vs watch orientation", "deg")
        orient.setLabel("bottom", "time (s)")

        self.curves["gyro_raw_dps"] = [
            raw.plot(pen=_pen(c), name=n) for c, n in zip(SERIES, "xyz", strict=True)
        ]
        self.curves["gyro_dps"] = [
            clean.plot(pen=_pen(c), name=n) for c, n in zip(SERIES, "xyz", strict=True)
        ]
        self.curves["knock_signal_g"] = [knock.plot(pen=_pen(SERIES[0]), name="signal")]
        knock.addItem(
            pg.InfiniteLine(knock_threshold_g, angle=0, pen=_pen(INK_SECONDARY, 1, dashed=True))
        )
        self.toggle_marks = knock.plot(
            pen=None, symbol="t", symbolBrush=SERIES[1], symbolPen=None, name="toggle"
        )
        self.curves["position_raw_px"] = [
            cursor.plot(pen=_pen(SERIES[0], 1, dashed=True), name="x integrated"),
            cursor.plot(pen=_pen(SERIES[1], 1, dashed=True), name="y integrated"),
        ]
        self.curves["position_smooth_px"] = [
            cursor.plot(pen=_pen(SERIES[0], 2), name="x smoothed"),
            cursor.plot(pen=_pen(SERIES[1], 2), name="y smoothed"),
        ]
        self.curves["tilt_error_deg"] = [orient.plot(pen=_pen(SERIES[0]), name="tilt error")]
        self.curves["heading_error_deg"] = [
            orient.plot(pen=_pen(SERIES[1]), name="heading difference")
        ]

        self.status = QtWidgets.QLabel()
        self.status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.status.setMinimumWidth(330)
        self.status.setStyleSheet(
            f"background:{SURFACE}; color:{INK}; padding:14px; font-family:Consolas; font-size:12px"
        )

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(plots, stretch=1)
        layout.addWidget(self.status)
        self.setCentralWidget(central)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(round(1000 / refresh_hz))

    def refresh(self) -> None:
        snap = self.telemetry.snapshot()
        if len(snap.time_s) > 0:
            for name, curves in self.curves.items():
                values = snap.signals[name]
                for column, curve in enumerate(curves):
                    # connect="finite" leaves gaps where values are NaN (invalid or disengaged).
                    curve.setData(snap.time_s, values[:, column], connect="finite")
            toggled = snap.signals["toggled"][:, 0] > 0
            self.toggle_marks.setData(
                snap.time_s[toggled], snap.signals["knock_signal_g"][toggled, 0]
            )
        self.status.setText(self._status_html(snap))

    def _status_html(self, snap: TelemetrySnapshot) -> str:
        s = snap.status
        engaged = bool(s.get("engaged", False))
        clutch_color = SERIES[2] if engaged else INK_SECONDARY
        clutch = "ENGAGED" if engaged else "off"
        rows = [
            f"<div style='font-size:20px; color:{clutch_color}'><b>Pointer {clutch}</b></div>",
            "<div style='color:#52514e'>double knock or hotkey to toggle</div><br>",
            f"toggles &nbsp;&nbsp;&nbsp;&nbsp; {s.get('toggles', 0)}",
            f"frames played {s.get('frames', 0)}",
            "<br><b>Playout buffer</b>",
            f"delay &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {self.playout_delay_s:.2f} s",
            f"buffered &nbsp;&nbsp; {float(s.get('buffered_s', 0.0)):.2f} s",
            f"waiting &nbsp;&nbsp;&nbsp; {'yes' if s.get('starved') else 'no'}",
            f"underruns &nbsp; {s.get('underruns', 0)}",
            "<br><b>Latency (ms)</b> &nbsp; p50 &nbsp;&nbsp; p95",
        ]
        latency = s.get("latency", {})
        assert isinstance(latency, dict)
        for key, label in LATENCY_ROWS:
            if key in latency:
                p50, p95 = latency[key]["p50"], latency[key]["p95"]
                rows.append(f"{label}<br>&nbsp;&nbsp;&nbsp;&nbsp;{p50:9.2f} {p95:8.2f}")
        return "<br>".join(rows)


def run_dashboard(window: Dashboard) -> None:
    """Show the window until it is closed or Ctrl+C is pressed in the console."""
    app = QtWidgets.QApplication.instance()
    assert app is not None, "create the QApplication before the Dashboard"
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # Qt's event loop runs in C++; a periodic no-op timer hands control back to Python so the
    # Ctrl+C handler above gets a chance to run.
    keep_alive = QtCore.QTimer()
    keep_alive.timeout.connect(lambda: None)
    keep_alive.start(200)
    window.show()
    app.exec()

