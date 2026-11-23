# Results

## Phase 0

**Status: method defined, waiting for real recordings.** No numbers below are assumed; every
table is filled in from recorded sessions.

### Recording protocol

Set up the network and app first ([SETUP_NETWORK.md](SETUP_NETWORK.md)). For each session,
start the probe with a descriptive note, then start streaming on the phone:

```powershell
python scripts/phase0_probe.py --note "S1 still on desk, watch rate <X> Hz, push period <Y>"
```

| # | Session | Duration | Purpose |
|---|---|---|---|
| S1 | Watch lying still on the desk | 2 min | Sample rate, noise floor, gyro bias |
| S2 | Wearing the watch, normal wrist motion | 2 min | Rate and jitter under motion |
| S3 | Wearing the watch, mixed activity | 10 min | Gaps over time, clock offset and drift |
| S4 | Repeat S1 at a second sample-rate setting (if the app offers one) | 2 min | Rate vs setting |
| S5 | (optional) SensorLog over UDP, as S2 | 2 min | Backup transport |

Afterwards, paste a few lines from one `packets.jsonl` file (or the whole file of a short
session) so the parser can be written from real data.

### Metrics (computed by `scripts/phase0_analyze.py`, written after inspecting real packets)

See [LEARNING.md](LEARNING.md#phase-0-understanding-the-stream) for explanations.

| Metric | Definition |
|---|---|
| Effective sample rate (per sensor) | From the sensor's own timestamps: median period and N / span |
| Sample-timestamp jitter | Histogram of differences between consecutive sample timestamps |
| Packet inter-arrival jitter | Histogram of differences between consecutive `recv_mono_ns` |
| Burstiness | Histogram of samples per packet |
| Gaps / loss | Periods > 1.5 x nominal: count, total duration, estimated missing samples |
| Clock offset and drift | `recv_wall - last_sample_time` per packet; lower envelope; linear fit for drift |

### Measured numbers

*To be filled in.*

### End-to-end latency with 240 fps video

This measures the full path from physical motion to something changing on screen.

1. Show something on screen that reacts immediately to a sharp acceleration spike. Phase 0
   can provide a small "flash" indicator once the parser exists; from Phase 2 the live
   dashboard is used instead.
2. Put the phone in **slow-motion video mode at 240 fps** (Camera -> Slo-mo; check 240 fps is
   selected in Settings -> Camera -> Record Slo-mo). Use a second phone if the streaming
   phone is busy, or use the streaming phone only if recording video doesn't disturb
   streaming.
3. Frame the shot so both the watch and the laptop screen are visible.
4. Tap the watch sharply on the desk ~20 times, a few seconds apart.
5. In the video, for each tap, count frames from the frame where the watch hits the desk to
   the first frame where the screen changes. Latency = frames / 240 s (one frame is about
   4.17 ms).
6. Report median and spread over all taps. Note the display refresh rate: a 60 Hz screen
   adds up to ~16.7 ms of quantisation on its own.

### Go / no-go criteria (proposed, to be confirmed)

| Criterion | Proposed threshold | Why |
|---|---|---|
| Watch IMU data arrives live | Yes | Without it the project needs a different data path |
| Gyroscope effective rate | >= 50 Hz | Pointer smoothness and gesture detail |
| Packet loss / gaps | < 1 % of time in gaps | Gaps cause cursor stalls and missed gestures |
| Median packet interval (batching) | <= 100 ms | Batching adds directly to perceived latency |
| End-to-end latency (video) | p50 <= 150 ms | Above this, pointing feels laggy |

The free Sensor Logger tier pushes once per second, so the two latency criteria can only be
met on the paid tier (100 ms push). Per [DECISIONS D-007](DECISIONS.md), Phase 0 measures the
free tier now, and the latency criteria are re-measured when the subscription is taken for
live testing.

### Recommendation

*Pending measurements.*
