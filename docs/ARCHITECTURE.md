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

## Current state (Phase 0)

Only the feasibility probe exists:

```
phone app --HTTP POST--> ProbeServers (ThreadingHTTPServer) --+
phone app --UDP-------->  ProbeServers (UDP socket thread)  --+--> RawPacketLog (JSONL, verbatim)
```

- `src/airhand/ingest/probe_server.py`: HTTP and UDP receivers that do not interpret payloads.
- `src/airhand/ingest/raw_log.py`: the packet "envelope" format and its JSONL reader/writer.
- `scripts/phase0_probe.py`: command-line entry point.
- `src/airhand/ingest/sensor_logger.py`: parser for Sensor Logger messages (written from
  real packets).
- `src/airhand/ingest/probe_session.py`: loads a recorded session into packet and
  per-sensor sample tables.
- `src/airhand/ingest/stream_stats.py` + `scripts/phase0_analyze.py`: stream measurements.

Measured transport (Phase 0): the watch hands samples to the phone in 51-sample (~0.51 s)
chunks; the phone pushes one HTTP packet every ~1.0-1.5 s on the free tier. See
[RESULTS.md](RESULTS.md#phase-0).

## Latency budget

To be filled in once real measurements exist (Phase 0 measures the network leg; later
phases add resample, DSP, inference and dispatch).

| Stage | Budget | Measured p50 | Measured p95 |
|---|---|---|---|
| Watch -> phone -> laptop (Sensor Logger free tier; sample age, +/-0.2 s clock uncertainty) | TBD | ~1.8 s | ~2.6 s |
| Resample | TBD | | |
| DSP | TBD | | |
| Inference | < 5 ms (p95) | | |
| Dispatch | TBD | | |
