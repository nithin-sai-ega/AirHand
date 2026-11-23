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

### D-007: Sensor Logger free tier for now; paid tier only when live latency matters
- **Finding:** in a first 15 s test, Sensor Logger's free tier pushed one HTTP packet per
  second (about 600 samples each, all streams at about 100 Hz). A shorter push interval
  (100 ms) requires a paid subscription.
- **Choice:** use the free tier for Phase 0 measurements and for all data collection,
  training and offline/replay evaluation (Phases 1, 3 and 4). Every sample carries its own
  timestamp, so batching does not affect the recorded data. Subscribe only when live
  low-latency streaming is needed (live pointer tests in Phase 2, live gestures in Phase 5)
  and re-measure the stream at that point.
- **Consequence:** until then, live end-to-end latency is dominated by the 1 s batching, so
  live pointer and gesture tests are deferred or run on replayed sessions.
- **Alternatives:** SensorLog over UDP (free; unclear whether the watch can still stream
  live, and watch logging is described as up to about 50 Hz); other paid apps; a native
  watchOS app (needs a Mac, which is out of scope).
- **Update after the full Phase 0 measurements:** the watch delivers data to the phone in
  fixed 51-sample (~0.51 s) chunks, adding ~0.5 s before the phone's push timer even
  starts. The paid tier may therefore not reach the latency needed for a cursor. The first
  thing to do on the paid tier is re-run the probe and check the chunk size (RESULTS.md,
  Phase 0 recommendation).

### D-008: Parse only the watch's "wrist motion" stream; keep Apple's processed values
- **Choice:** the parser is generic (any sensor name, any value fields), but the pipeline
  uses `wrist motion`: rotation rate (rad/s), user acceleration and gravity (g), quaternion
  and roll/pitch/yaw.
- **Why:** it is the only watch stream Sensor Logger provides. It is Apple's fused
  device-motion output, so gyro bias is already removed (measured bias ~0.003 deg/s) and an
  orientation estimate is included, which later phases can compare against Madgwick.
- **Consequence:** we never see raw accelerometer/gyroscope readings from the watch; the
  "raw" input to our DSP is already processed by Apple.
- **Alternatives:** none available from this app.

### D-009: Analysis settings that depend on the recordings live in config
- **Choice:** which sessions form the report, the undisturbed window used for the noise
  estimate (S1, 35-85 s) and the start-up span skipped in the delay fit (15 s) are in
  `configs/phase0_analysis.yaml`.
- **Why:** they are judgments about specific recordings (the watch was bumped at 22-30 s;
  each recording starts by flushing a backlog), so they should be visible and editable,
  not hidden in code.
- **Alternatives:** automatic stillness detection (better later, in Phase 2, where it is
  needed anyway); robust statistics such as the median absolute deviation.
