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

## Phase 2: DSP and the pointer (no ML)

### Why the position is never computed from acceleration
It is tempting to get hand position by integrating acceleration twice. In practice it
fails within seconds. Every accelerometer has a small error (bias, noise, a slightly wrong
gravity estimate). Integrating once turns a constant error `b` into a velocity error `b*t`;
integrating again turns it into a position error `b*t^2/2`. With a tiny bias of 0.01 g,
after 5 s the position is already off by about 1.2 m, and the error keeps growing with the
square of time. So the pointer uses **rotation rate** (one integration, and only of what
the user is actively doing), and acceleration is only used for things that do not need
integration: the direction of gravity, and sharp knock spikes.

### Gyro bias and stillness
A gyroscope at rest still reports a small rate: its bias. If it were fed to the pointer, the
cursor would creep on its own. The bias can only be measured when the true rate is known to
be zero, so the code first detects stillness (rotation below 1 deg/s and acceleration below
0.02 g for 0.5 s), then moves the bias estimate towards the measured rate with a running
average. During motion the estimate is frozen. Apple's Core Motion already removes the bias,
so on the watch the estimate stays near zero (0.02 deg/s); the stage matters for a raw IMU.

### Low-pass filters (Butterworth)
Deliberate hand motion lives below roughly 10 Hz; anything faster on the gyro is mostly
sensor noise. A low-pass filter lets slow changes through and removes fast ones. Butterworth
filters are used because their pass band is flat, so slow motion keeps its exact size. The
filter is **causal** (uses only past samples, so it can run live) and keeps its internal
state between packets, so chunked and one-shot processing give identical output. The cost of
any causal filter is a small delay, which is why the cutoff is set well above the motion
band (15 Hz cutoff, about 15 ms of delay for hand-motion frequencies).

### Orientation: the Madgwick filter
The gyro alone gives orientation by integrating rotation: very accurate over a short time,
but small errors add up (drift). The accelerometer alone gives tilt (which way is down)
whenever the wrist is not accelerating, but it is noisy and says nothing about heading.
Madgwick combines them: every step it integrates the gyro, then takes one small
gradient-descent step that rotates the estimate so its "down" agrees with the measured
gravity. The step size `beta` sets the balance: small = trust the gyro, large = trust the
accelerometer. Heading cannot be corrected without a magnetometer, so it slowly drifts.

### The One Euro filter
A pointer has two conflicting needs: no jitter when aiming slowly, no lag when moving fast.
A fixed low-pass filter can only trade one against the other. The One Euro filter is a
low-pass whose cutoff frequency rises with speed (`cutoff = min_cutoff + beta * |speed|`):
heavy smoothing when the cursor is almost still, almost none during a fast sweep. It is
applied to the integrated cursor position, which is what it was designed for.

### Transfer functions (control-display gain)
"Gain" is how many pixels the cursor moves per degree of wrist rotation. A constant gain
forces a choice between precision and speed. Like every desktop OS's pointer acceleration,
the gain here grows smoothly with speed (a sigmoid from 3 to 20 px/deg): slow, careful
motion is precise, fast flicks cross the screen. A soft dead zone subtracts 1.5 deg/s from
the speed so tremor and noise at rest do not move the cursor, without a jump when motion
starts.

### Measuring rotation about gravity, not about the watch
The gyro measures rotation about the watch's own axes, which depend on how the forearm is
rolled. Projecting the rotation onto "up" (opposite to gravity, which the watch reports) and
onto a horizontal "left" axis makes "turn the hand left" mean "cursor left" whatever the
wrist roll. Rolling the forearm about its own length then barely moves the cursor at all.

### Playout (jitter) buffers, and why not predict instead
Packets arrive about once per second, each holding a second of samples. Applying a packet at
once would make the cursor jump once per second. Every sample does arrive, just late, so
each frame is played at a fixed time after it was recorded: its timestamp, plus the
fastest network delivery seen recently (the "floor"), plus a safety margin. Frames then come
out at their recorded 10 ms spacing and the cursor moves smoothly, about 3 s behind the hand
on the free tier. Video and voice calls do the same to hide network jitter.

The margin has to cover the *oldest* sample in a packet: a 1.5 s packet is sent when its
newest sample exists, so its first sample is already 1.5 s old on arrival. The floor is a
minimum over the last 30 s rather than over the whole run because the watch's timestamps and
the laptop clock tick at slightly different rates (clock drift); a schedule fixed at the
start would slowly drift until the buffer runs dry.

Prediction (extrapolating where the hand will be) only works for short horizons, tens of
milliseconds, because hand motion is not predictable over half a second: a prediction 0.5-1 s
ahead would be wrong whenever the hand changes direction. So prediction cannot remove this
delay; it can only shave off a little once the delay is already small (Phase 6c).

### Clutching and the double knock
A clutch decides when wrist motion moves the cursor, like lifting a mouse off the desk. It is
currently the F8 key as a toggle (press once for on, once for off). An optional detector
(off by default) toggles it by knocking the knuckles on the desk twice. A knock is a short, sharp spike in
acceleration, so the signal is high-pass filtered (slow arm motion removed) and thresholded.
Timing rules then turn spikes into a decision: two knocks 0.12-0.5 s apart, nothing just
before or just after, and a cooldown after each toggle. This is classic event detection
with hand-set rules. Apple's own pinch gesture works on the same principle but with a trained
model, which is what Phases 3-5 aim to reproduce from the motion data.

### Absolute vs relative cursor moves
A relative `SendInput` move goes through the Windows pointer-speed and "Enhance pointer
precision" settings, which would add a second, unknown gain curve. Reading the cursor
position, adding our move and sending an **absolute** position avoids that, so the transfer
function above is the only one applied.

### Qt and pyqtgraph
Qt is a toolkit for desktop windows (buttons, labels, layouts). pyqtgraph draws plots inside
a Qt window and is fast enough to redraw at 30 frames per second. A window must be drawn by
the program's main thread, so the network receiver and the pointer loop run in background
threads and the window only reads a snapshot of their data.
