# Architecture

## Data flow

```
Apple Watch (IMU) -> iPhone app -> Wi-Fi (HTTP/MQTT/UDP)
   -> [ingest]   async server, parser, timestamp-based resampler, ring buffer, recorder
   -> [dsp]      calibration, gyro-bias estimation, filters, orientation (Madgwick/EKF)
   -> +- [pointer]  gyro-rate -> transfer function -> One Euro filter -> clutch logic
      +- [gesture]  sliding window -> streaming TCN (ONNX Runtime) -> thresholds/hysteresis/cooldown
   -> [dispatch] state machine (IDLE / POINTING / DRAGGING / SCROLLING) -> Windows SendInput
   -> [telemetry] live dashboard: raw signals, class probabilities, per-stage latency
```

Every stage can be driven from recorded sessions (record/replay) so the pipeline stays
deterministic and testable without the watch.

## Current state (Phase 1)

```
Sensor Logger --HTTP POST--> SensorLoggerReceiver (aiohttp)  --+
                                                                +--> asyncio.Queue[SampleChunk]
recorded session ----------> replay() (real time or fast)   --+            |
                                                                           v
                                                  IngestPipeline.process(chunk)
                                                    |-> SessionRecorder  (raw samples -> Parquet parts + meta.json)
                                                    |-> Resampler        (99.9 Hz sensor time -> exact 100 Hz grid, gaps marked)
                                                    '-> RingBuffer       (last 30 s of frames for downstream stages)
```

The live receiver and the replayer put the same `SampleChunk` objects on the same queue, so
everything after the queue cannot tell live data from a recording.

| Module | Role |
|---|---|
| `ingest/sensor_logger.py` | Parse Sensor Logger JSON (written from real packets); extract the watch stream as a chunk |
| `ingest/chunk.py` | `SampleChunk` (raw samples of one packet) and `Frames` (grid samples + validity) |
| `ingest/http_server.py` | Async receiver: timestamp, parse, queue |
| `ingest/resampler.py` | Causal, streaming linear/nlerp resampler with gap marking |
| `ingest/ring_buffer.py` | Fixed-size buffer of recent frames |
| `ingest/recorder.py` | Session folders: Parquet part files + metadata; reader |
| `ingest/replayer.py` | Feeds recorded packets into the queue with original or scaled timing |
| `ingest/pipeline.py` | Connects recorder, resampler and ring buffer; frame digest for determinism |
| `scripts/ingest_live.py`, `replay_session.py`, `convert_probe_session.py` | Entry points |

Phase 0 tools (`probe_server.py`, `raw_log.py`, `probe_session.py`, `stream_stats.py`,
`phase0_probe.py`, `phase0_analyze.py`) remain for re-measuring the stream.

Measured transport (Phase 0): the watch hands samples to the phone in 51-sample (~0.51 s)
chunks; the phone pushes one HTTP packet every ~1.0-1.5 s on the free tier. See
[RESULTS.md](RESULTS.md#phase-0).

## Pointer (Phase 2)

```
 [ingest thread: asyncio]                       [pointer thread: 100 Hz ticks]
 HTTP -> IngestPipeline --on_frames--> PlayoutBuffer --pop_due(now)--> DspProcessor
          (recorder, resampler)        (time + floor + delay)            | bias (while still)
                                                                         | low-pass gyro
                                                                         | Madgwick (compare)
                                                                         v
 F8 (GetAsyncKeyState) ----------------------------------------> PointerProcessor
                                                                   | gravity-referenced mapping
                                                                   | gain curve (dead zone)
                                                                   | integrate -> One Euro -> round
                                                                   | double knock -> clutch
                                                                   v
                                                      WindowsMouse (absolute SendInput)
                                                                   |
 [main thread: Qt] Dashboard <-- 30 Hz snapshot -- TelemetryBuffer + StageTimer
```

| Module | Role |
|---|---|
| `ingest/playout.py` | Jitter buffer: plays each frame at its time + network floor + delay, following clock drift |
| `ingest/live.py` | `LiveIngest` (receiver + pipeline, explicit start/stop) and `IngestThread` |
| `dsp/quaternion.py` | Quaternion helpers (Core Motion convention: body -> world) |
| `dsp/filters.py` | Streaming Butterworth filters; gap-aware wrapper that never filters across invalid frames |
| `dsp/stillness.py` | Stillness detector and gyro bias estimator |
| `dsp/madgwick.py`, `dsp/one_euro.py` | Orientation filter; adaptive low-pass for the cursor |
| `dsp/processor.py` | DSP stage over a batch of frames |
| `pointer/mapping.py`, `pointer/transfer.py` | Rotation about gravity-defined axes; gain curve |
| `pointer/clutch.py` | Double-knock detector and clutch state |
| `pointer/processor.py` | Pointer stage: per-frame cursor moves, knock toggles |
| `pointer/engine.py` | The live tick loop |
| `dispatch/windows_input.py` | Absolute SendInput cursor moves, hotkey polling, a recording stand-in |
| `ui/telemetry.py`, `ui/dashboard.py`, `latency.py` | Dashboard data, window and per-stage timing |
| `scripts/pointer_live.py`, `pointer_replay.py`, `compare_orientation.py` | Entry points |

Live and replayed data take the same path: `pointer_replay.py --live` feeds recorded packets
to the pipeline with their original arrival timing, and an engine test with a fake clock
checks that the live loop moves the cursor exactly as offline processing does.

## Latency budget

Filled in as stages are built and measured.

| Stage | Budget | Measured p50 | Measured p95 |
|---|---|---|---|
| Watch -> phone -> laptop (Sensor Logger free tier; sample age, +/-0.2 s clock uncertainty) | TBD | ~1.8 s | ~2.6 s |
| Parse to ring buffer (resample + buffer, per ~1 s packet, Intel i7) | TBD | 0.11 ms | 0.18 ms |
| Playout margin (so ~1-1.5 s packets play smoothly) | `playout_delay_s` | 2.5 s (set) | 2.5 s |
| DSP (per 10 ms tick) | < 2 ms | 0.12 ms | 0.28 ms |
| Pointer (per tick) | < 2 ms | 0.21 ms | 0.48 ms |
| Inference | < 5 ms (p95) | | |
| Dispatch: GetCursorPos + SendInput (per tick) | < 2 ms | 0.32 ms | 0.63 ms |

On the free tier the pointer is about 3.2 s behind the hand (newest-sample age floor ~0.75 s
plus the 2.5 s playout margin); the laptop-side processing adds well under 2 ms. The transport, not the computation,
dominates the budget.
