# AirHand

Control a Windows laptop with an Apple Watch. Wrist rotation moves the cursor, and a
streaming neural network recognises discrete gestures (flick, twist, double tap, circle,
shake) to trigger actions. Everything runs on the laptop CPU, and model size, INT8
quantization and latency are measured as part of the project.

> Status: **Phase 1, ingest.** The watch stream is received, resampled onto a 100 Hz grid,
> buffered, recorded to Parquet and replayable deterministically. Phase 0 measured the
> stream at 99.9 Hz with no gaps or loss, delivered in ~1 s batches
> (see [docs/RESULTS.md](docs/RESULTS.md)).

## Pipeline (planned)

```
Apple Watch (IMU) -> iPhone app -> Wi-Fi (HTTP/UDP)
  -> ingest -> dsp -> pointer (DSP, no NN)      -> dispatch -> Windows SendInput
                   -> gesture (streaming TCN)  ->
  -> telemetry dashboard
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Setup (Windows, Python 3.11+)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Phase 0: run the probe

1. Follow [docs/SETUP_NETWORK.md](docs/SETUP_NETWORK.md) to open the firewall ports and
   configure the phone app.
2. Start the probe:

   ```powershell
   python scripts/phase0_probe.py --note "still on desk, 100 Hz"
   ```

3. Start streaming on the phone. Every packet is saved verbatim to
   `data/raw/probe/<timestamp>/packets.jsonl`. Press `Ctrl+C` to stop.
4. List the sessions to analyse in `configs/phase0_analysis.yaml`, then:

   ```powershell
   python scripts/phase0_analyze.py
   ```

## Record and replay sessions

1. Set up the network and phone app ([docs/SETUP_NETWORK.md](docs/SETUP_NETWORK.md)); the
   push URL is `http://<laptop-ip>:8000/data`.
2. Record a live session (Ctrl+C to stop):

   ```powershell
   python scripts/ingest_live.py --note "desk test" --subject S01 --wrist L
   ```

   Sessions are saved to `data/raw/sessions/<timestamp>/` (Parquet parts + `meta.json`).
3. Replay a session through the same pipeline (`--speed 1` for real time, default is as
   fast as possible); the printed digest is identical on every run:

   ```powershell
   python scripts/replay_session.py data/raw/sessions/<timestamp>
   ```

4. Phase 0 probe recordings can be converted into sessions:

   ```powershell
   python scripts/convert_probe_session.py data/raw/probe/<timestamp>
   ```

Settings live in `configs/ingest.yaml`.

## Development

```powershell
ruff check .
pytest
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): system design and latency budget
- [docs/DECISIONS.md](docs/DECISIONS.md): design decisions and alternatives
- [docs/LEARNING.md](docs/LEARNING.md): plain-English explanations of each technique
- [docs/RESULTS.md](docs/RESULTS.md): measurements and model results
- [docs/SETUP_NETWORK.md](docs/SETUP_NETWORK.md): firewall, hotspot and phone-app setup
