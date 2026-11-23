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

## Latency budget

Filled in as stages are built and measured.

| Stage | Budget | Measured p50 | Measured p95 |
|---|---|---|---|
| Watch -> phone -> laptop (Sensor Logger free tier; sample age, +/-0.2 s clock uncertainty) | TBD | ~1.8 s | ~2.6 s |
| Parse to ring buffer (resample + buffer, per ~1 s packet, Intel i7) | TBD | 0.11 ms | 0.18 ms |
| DSP | TBD | | |
| Inference | < 5 ms (p95) | | |
| Dispatch | TBD | | |
