# Learning notes

Plain-English explanations of the techniques used in this project, phase by phase.

## Phase 0: understanding the stream

### Why look at raw packets first?
We do not control the phone app's packet format. If a parser is written from guesses, a
wrong field name or unit fails silently and corrupts everything downstream. So the probe
stores every packet byte-for-byte, and the parser is written only after reading real ones.

### Wall clock vs monotonic clock
- **Wall clock** (`time.time_ns`) is "what time is it". The OS can move it forwards or
  backwards when it syncs with a time server. It is needed to compare against timestamps
  generated on the phone or watch.
- **Monotonic clock** (`time.perf_counter_ns`) only ever moves forwards at a steady rate
  and has high resolution. It is the right clock for measuring *intervals*, such as the
  time between two packets.

### Sample rate, jitter, burstiness, gaps
- **Effective sample rate:** how many samples per second actually arrive, computed from
  the sensor's own timestamps. It can differ from the rate requested in the app.
- **Jitter:** variation in the spacing between samples (or packets). A perfect 100 Hz
  stream has every gap equal to 10 ms; real streams wobble around that.
- **Burstiness:** apps often batch samples, e.g. 20 samples in one packet every 200 ms.
  Data then arrives in lumps, which adds latency (a sample waits for its batch) even if
  the average rate is fine.
- **Gaps / packet loss:** stretches where samples are missing. A gap is flagged when the
  spacing is much larger than normal (e.g. > 1.5 x the nominal period).

### Clock offset
The watch/phone and the laptop each have their own clock. For each packet,
`receive_time - sample_time` equals *clock offset + network delay*. Network delay varies but
never goes below some minimum, so the smallest values over a window (the "lower envelope")
estimate *offset + minimum delay*. If that envelope slopes over a long recording, the two
clocks run at slightly different rates (clock **drift**).
