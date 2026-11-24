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

## Phase 1

### D-010: aiohttp for the live HTTP receiver
- **Choice:** `aiohttp` (added as a dependency, approved in the Phase 1 plan).
- **Why:** the standard library has no async HTTP server. Packets arrive about once a second,
  so performance is irrelevant; a well-known, maintained server is easier to trust than
  hand-written HTTP parsing.
- **Alternatives:** HTTP parsing on `asyncio.start_server` (no dependency, more code to
  maintain); the threaded standard-library server from Phase 0 feeding a queue.

### D-011: Linear interpolation onto an exact 100 Hz grid; nlerp for quaternions
- **Choice:** grid points at exact multiples of 10 ms of the sensor clock; channels are
  interpolated linearly, the quaternion with normalised linear interpolation (with sign
  alignment); roll/pitch/yaw are recorded but not resampled.
- **Why:** the watch runs at 99.9 Hz, so its samples slowly slide relative to a 100 Hz grid;
  a fixed grid makes windows, filters and models independent of that. Linear interpolation
  never overshoots, needs only the two neighbouring samples (no lookahead, so no added
  latency), and its error is tiny at 100 Hz (bounded and tested). Euler angles wrap at
  +/-180 degrees, where interpolation produces nonsense.
- **Alternatives:** zero-order hold (repeat the last value; adds steps and up to 10 ms of
  timing error); keeping the native 99.9 Hz (windows would drift against wall time);
  higher-order or band-limited interpolation (needs future samples, adds latency).

### D-012: Gaps longer than 50 ms are marked invalid, never filled
- **Choice:** if neighbouring samples are more than `max_interp_gap_ms` (50 ms = 5 missing
  samples) apart, grid points between them get `valid = False` and NaN values.
- **Why:** filled-in data looks real to every later stage (a model could "detect" a gesture
  in invented data). Short gaps are interpolated because they are harmless at 100 Hz.
  Phase 0 measured no gaps, so this is a safety net.
- **Alternatives:** interpolate everything (silent corruption); drop invalid frames (breaks
  the fixed grid that windows rely on).

### D-013: Record raw samples as Parquet part files, with packet timing
- **Choice:** each session folder holds `meta.json` and `samples/part-NNNNN.parquet`, one
  row per raw sample with `packet_seq`, `recv_wall_ns`, `recv_mono_ns`, `time_ns` and all 16
  wrist motion fields. A new part file is written on every flush (5 s).
- **Why:** raw data can always be resampled again, deterministically. Packet numbers and
  receive times let replay reproduce the real packet timing. A Parquet file is unreadable
  until it is closed, so separate complete parts mean a crash loses at most 5 s, not the
  whole session (the first version kept one file open; this was caught while planning the
  live test and fixed, with a crash test).
- **Alternatives:** one Parquet file (unsafe on crash); CSV (larger, slower, no types);
  recording resampled frames (loses the original timing).

### D-014: Explicit start/stop for the live process
- **Choice:** `ingest_live.py` starts the server and pipeline, waits until Ctrl+C/Ctrl+Break,
  then runs a separate `stop()` that drains the queue and completes the metadata. Ctrl+Break
  is mapped to the same handler as Ctrl+C on Windows.
- **Why:** an end-to-end test showed that cleanup placed in a `finally` inside the
  long-running coroutine did not run when the loop was interrupted by Ctrl+Break. A
  separate stop step on the same event loop runs in every case.
- **Alternatives:** loop signal handlers (`add_signal_handler` is not supported on Windows).

## Phase 2

### D-015: Adaptive playout (jitter) buffer instead of applying packets immediately
- **Choice:** each frame is played at `sensor time + floor + playout_delay_s`, where the
  floor is the smallest (arrival time - newest sample time) over the last
  `playout_offset_window_s` (30 s). Delay 2.5 s. A packet later than the margin allows is
  counted as an underrun and its overdue frames are played at once (catch-up).
- **Why:** packets arrive about once per second (D-007), so applying them directly makes
  the cursor jump once per second; every sample does arrive, so smoothing costs only delay.
  The delay must cover the oldest sample of a packet (packets hold up to 1.53 s) plus jitter;
  on the Phase 0 sessions 2.5 s covered the p95 lateness (sweep in RESULTS.md). A sliding
  floor follows clock drift between watch timestamps and the laptop clock (about 1 ms/s
  measured in Phase 0). A test with that drift shows a fixed floor stalling repeatedly
  while the sliding one does not.
- **History:** the first version fixed the schedule at the first packet and skipped ahead on
  backlogs. A live smoke test showed its backlog rule cutting normal 1.5 s packets, and a
  replay of the real arrival times still showed periodic stalls from drift. Both led to this
  design.
- **Alternatives:** apply bursts immediately (jumpy); extrapolate between packets (wrong
  whenever the hand changes direction, see LEARNING.md); time-stretch playback to absorb
  jitter (smoother than catch-up bursts, more complex; worth it once delays are small).

### D-016: Pointer axes from gravity, not from the watch's own axes
- **Choice:** horizontal cursor speed = rotation about "up" (minus gravity); vertical =
  rotation about the horizontal axis perpendicular to the forearm (`up x forearm_axis`).
- **Why:** independent of wrist roll (tested: the same world motion at 0, 35, 90 and -60
  degrees of roll gives the same cursor motion), and wrist roll alone does not move the
  cursor. The watch reports gravity directly, so the pointer path needs no orientation filter.
- **Alternatives:** raw gyro axes (simpler, breaks when the forearm rolls); absolute
  orientation angles as the pointer position (needs heading, which drifts without a
  magnetometer).

### D-017: Gain curve, then One Euro on the integrated position, then rounding
- **Choice:** angular velocity -> sigmoid gain curve with a soft dead zone -> velocity
  integrated into a floating-point position -> One Euro filter -> rounded; the cursor move is
  the change in the rounded position.
- **Why:** this is the order in the project brief, and filtering *position* is how the One
  Euro filter is meant to be used (its speed estimate is then the cursor speed). Rounding the
  position rather than each step keeps sub-pixel motion (tested). The integrator and filter
  reset whenever the clutch disengages, so re-engaging never jumps.
- **Alternatives:** One Euro on the angular velocity before the gain (its adaptation would
  key off acceleration rather than speed); constant gain (precision vs speed trade-off).

### D-018: Double knock on the desk as the clutch, F8 as backup (knock later disabled)
- **Choice (owner's decision):** the pointer starts disengaged; a double knock (high-passed
  acceleration above 0.6 g twice, 0.12-0.5 s apart, quiet before and after, 1 s cooldown)
  or the F8 key toggles it.
- **Why:** hands-free, and a knock is a strong, short spike that plain DSP can separate from
  arm motion. The watch's own finger-pinch "double tap" is not available through Sensor
  Logger. This changes the planned roles of the v1 gestures (`circle` was the clutch,
  `double_tap` something else); the final roles are confirmed in Phase 3.
- **Alternatives:** a key only (the brief's stand-in; awkward with one hand on the watch); a
  raise-wrist-to-point posture (no gesture needed, but talking with the hands would move the
  cursor).
- **Open:** the threshold is provisional until a recording with deliberate knocks exists.
  Everyday motion in the Phase 0 sessions reached 0.88 g at most and produced no double
  knocks.
- **Update (owner's decision, after trying the idea):** the knock felt awkward. It is now
  off by default (`knock_enabled: false`) and the clutch is the F8 key as a toggle: one press
  turns the pointer on, the next turns it off, nothing is held. What the owner actually
  wanted is the watch's own AssistiveTouch pinch / clench gesture; Sensor Logger does not
  pass that event on, so `pinch` and `clench` become candidate gestures for our own model
  (Phase 3 data collection decides whether the motion sensors alone can separate them).
  The knock code and its tests stay, since it costs nothing when disabled.

### D-022: Point with the whole arm via rotation; no hand-position tracking
- **Choice:** the pointer follows rotation, whether it comes from the wrist, the elbow or the
  shoulder. Sweeping the arm left/right turns the watch about the vertical axis and moves the
  cursor; up/down comes from tilting the hand/forearm.
- **Why:** this is how VR controller "laser pointers" aim too (orientation, not position).
  Pure sideways sliding of the hand without turning would need position, and position from
  an accelerometer alone drifts by metres within seconds (LEARNING.md); headsets use cameras
  for it.
- **Alternatives:** an absolute "laser pointer" mode (arm direction -> screen position)
  instead of relative motion; possible later, but its left/right reference drifts in steps
  (RESULTS.md, Phase 2), so it would need re-centring with the clutch.

### D-019: Absolute SendInput moves, per-monitor DPI aware
- **Choice:** read the cursor position, add the move, clamp to the virtual desktop, send an
  absolute SendInput move normalised to 0..65535.
- **Why:** relative moves pass through Windows pointer acceleration, which would stack an
  unknown second gain curve on ours. DPI awareness keeps the coordinates in real pixels.
- **Alternatives:** relative SendInput (simpler, double acceleration); `SetCursorPos`
  (moves the cursor without generating a mouse input event); `pynput` (an extra dependency
  wrapping the same call).

### D-020: Ingest on its own event loop, a pointer loop thread, Qt in the main thread
- **Choice:** `IngestThread` runs the aiohttp receiver and pipeline on a private asyncio
  loop; `PointerEngine` ticks at 100 Hz in another thread; the dashboard reads a locked
  telemetry snapshot from the main thread at 30 Hz. The playout buffer, stage timer and
  telemetry are the only shared objects and each takes a lock.
- **Why:** Qt must own the main thread; a separate pointer loop means a slow redraw can never
  delay a cursor move. The Phase 1 ring buffer is not read across threads, so it stays
  lock-free (the Phase 1 report expected to lock it; not needed).
- **Alternatives:** `qasync` to merge asyncio into Qt's loop (extra dependency, and a slow
  redraw would then also block the network and the pointer).

### D-021: DSP settings tuned on the Phase 0 recordings
- **Choice:** stillness threshold 1.0 deg/s (was 2.0) and Madgwick beta 0.01 (was 0.04).
- **Why:** at 2 deg/s, slow real motion counted as stillness and pulled the bias estimate to
  0.17 deg/s; at 1 deg/s the still desk recording is still 91 % still and the estimate stays
  at or below 0.02 deg/s (noise floor about 0.1 deg/s). A beta sweep against the watch's own
  tilt gave the lowest error at 0.01 (p95 0.41 deg vs 0.67 at 0.04). The reference is
  Apple's fused attitude, not ground truth.
- **Alternatives:** keep textbook defaults; retune once more varied recordings exist.
