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

## Latency budget

To be filled in once real measurements exist (Phase 0 measures the network leg; later
phases add resample, DSP, inference and dispatch).

| Stage | Budget | Measured p50 | Measured p95 |
|---|---|---|---|
| Watch -> phone -> laptop (network) | TBD | TBD | TBD |
| Resample | TBD | | |
| DSP | TBD | | |
| Inference | < 5 ms (p95) | | |
| Dispatch | TBD | | |
