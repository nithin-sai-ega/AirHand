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

Drift is quoted in **ppm** (parts per million): 1 ppm means the clocks separate by 1 us
every second, or 3.6 ms per hour. Quartz clocks in phones and laptops are typically within
tens of ppm. So when the Phase 0 fit gave +953 ppm, that was a sign the line was fitting
something else (the delay switching between two levels), not real clock drift.

### Two ways to estimate a sample rate
- **From the median spacing:** robust to gaps; tells you the rate the sensor *runs* at.
- **From count / duration:** includes gaps; tells you the rate actually *delivered*.

If they agree, nothing is missing. In Phase 0 both were 99.9 Hz.

### Batching vs delay: the freshest and the oldest sample
When an app sends samples in batches, the samples in one packet have different ages on
arrival. The **oldest** sample waited for the whole batch to fill. The **freshest** sample
waited for almost nothing, so its age is the pure transport delay (plus clock offset).
Comparing the two separates "waiting for the batch" from "travelling over the network".

Comparing two streams in the **same** recording (phone gyroscope vs watch) cancels the clock
offset, because both are compared against the same laptop clock. That is how Phase 0
isolated the ~0.5 s added by the watch-to-phone link.

### Why "too perfect" timestamps are a warning sign
Real sensor clocks jitter by microseconds. The watch timestamps were exactly 10.01 ms apart
every time, which means they were most likely *computed* (start time + n x period) rather
than captured per sample. Such timestamps are fine for a uniform grid, but they cannot be
used to measure the sensor's true jitter, and they hide any real sampling irregularity.

### Device motion vs raw IMU
Apple's Core Motion offers raw sensor readings and a fused "device motion" output. Device
motion combines the accelerometer and gyroscope (sensor fusion) to give: rotation rate with
bias removed, gravity direction, user acceleration (total acceleration minus gravity) and
an orientation quaternion. The watch stream we get is device motion, which is why the
measured gyro bias at rest was essentially zero.

## Phase 1: ingest, resampling, record/replay

### Why resample at all?
Filters, sliding windows and neural networks assume samples are evenly spaced: "the last
50 samples" should always mean "the last 0.5 s". The watch's timestamps are 10.01 ms apart
(99.9 Hz), so after 100 s the sample count and the time have drifted apart by 0.1 s.
Resampling puts every value on a fixed grid (every 10 ms exactly) using the sensor's own
timestamps, so time and sample index always agree.

### Linear interpolation, and how wrong it can be
To get a value at grid time g between samples at t0 and t1:
`alpha = (g - t0) / (t1 - t0)`, `value = v0 + alpha * (v1 - v0)`.
For a smooth signal the error is at most `h^2 / 8 * max|f''|`, where h is the spacing
between samples. At 100 Hz with a 2 Hz sine (fast wrist motion) that is about 0.2 % of the
amplitude. The tests check this bound on a jittered sine.

### Causal processing
The resampler only uses samples that have already arrived; it never waits for future ones
and never extrapolates past the newest sample. That matters for real time: any method that
needs future samples adds their duration to the latency.

### Why long gaps are not filled
Interpolating across 10 missing samples would draw a straight line through 100 ms of motion
that was never measured. Downstream stages would treat it as real. Marking those frames
invalid keeps the grid regular while making the missing data explicit.

### Quaternions
A unit quaternion (w, x, y, z) describes a 3-D rotation, and q and -q describe the same
rotation. To interpolate: if the two quaternions point "away" from each other
(dot product < 0), flip one; then interpolate linearly and rescale to length 1 (nlerp).
Slerp (spherical interpolation) is exact, but for rotations of a degree or two between
samples 10 ms apart, nlerp gives practically the same result and is simpler.

### Ring buffers
A fixed-size array with a write index that wraps around. New data overwrites the oldest,
memory never grows, and nothing is allocated per sample. Reading the last n items means
reading backwards from the write index, wrapping past the start if needed.

### Determinism and digests
The pipeline hashes every frame it produces (SHA-256). Same input -> same bytes -> same
digest. Comparing two 64-character strings is an easy way to prove that replay is
deterministic, and that packet boundaries or replay speed do not change the output.

### Parquet and crash safety
Parquet is a columnar file format: each column is stored (and compressed) separately, so
reading one channel does not read the others. Its index ("footer") is written when the file
is closed, so an open file is not readable yet. Writing a new small file every few seconds
means a crash only loses the data since the last flush.
