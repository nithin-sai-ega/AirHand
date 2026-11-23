# Design decisions

Each entry records a non-obvious choice, why it was made, and what else was considered.

## Phase 0

### D-001: Python 3.13 in a local venv
- **Choice:** `py -3.13 -m venv .venv`; the package declares `requires-python >= 3.11`.
- **Why:** 3.13 is already installed. Before choosing it, PyPI was checked for Windows
  x86-64 wheels of the packages planned for later phases: `torch`, `onnxruntime`,
  `onnxruntime-openvino`, `openvino` and `scikit-learn` all ship `cp313-win_amd64` wheels,
  and `qai-hub`, `mlflow`, `pyqtgraph` and `xgboost` are pure-Python or ABI-independent.
- **Alternatives:** Python 3.12 (would need an extra install for no measured benefit);
  the system 3.9 on PATH (too old: `onnxruntime` and `torch` now require 3.11+).

### D-002: Standard-library servers for the probe
- **Choice:** `http.server.ThreadingHTTPServer` and a plain UDP socket on a thread.
- **Why:** the probe has a single job, recording exactly what arrives, so it should be
  small enough to trust at a glance and have no dependencies. The real ingest server
  (Phase 1) is where async design matters.
- **Alternatives:** an `asyncio` server (better for Phase 1, unnecessary here);
  `aiohttp` (an extra dependency outside the core stack).

### D-003: JSONL "envelope" per packet, body stored verbatim
- **Choice:** one JSON line per packet with receive timestamps, transport, peer, HTTP
  method/path/headers, and the body. The body is stored as text if it is valid UTF-8,
  otherwise base64; both round-trip to the identical bytes.
- **Why:** the payload format is not ours, so nothing is parsed at capture time. JSONL is
  appendable, greppable, and easy to replay.
- **Alternatives:** one `.bin` file per packet (harder to inspect and replay); parsing to a
  table at capture time (would bake in guesses about the schema).

### D-004: Two receive clocks
- **Choice:** each packet records `time.time_ns()` (wall clock) and `time.perf_counter_ns()`
  (monotonic), taken right after the full body has been read.
- **Why:** wall time is needed to compare with the watch/phone timestamps (clock offset);
  the monotonic clock is immune to NTP adjustments and is the right one for intervals
  (jitter, inter-arrival times).
- **Alternatives:** wall clock only (intervals can jump when the OS adjusts the clock).

### D-005: matplotlib for static report plots
- **Choice:** histograms and time-series plots in `docs/` are made with matplotlib.
- **Why:** it is the standard tool for static figures; pyqtgraph (kept for the live
  dashboard) is designed for interactive real-time plots and is awkward for report images.
- **Alternatives:** the pyqtgraph image exporter.

### D-006: Default ports HTTP 8000, UDP 5005
- **Choice:** configurable in `configs/probe.yaml`.
- **Why:** common, unprivileged ports that rarely clash with other software.
