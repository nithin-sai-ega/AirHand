# AirHand

Control a Windows laptop with an Apple Watch. Wrist rotation moves the cursor, and a
streaming neural network recognises discrete gestures (flick, twist, double tap, circle,
shake) to trigger actions. Everything runs on the laptop CPU, and model size, INT8
quantization and latency are measured as part of the project.

> Status: **Phase 2, pointer (no ML).** Wrist rotation moves the Windows cursor through
> gyro bias removal, filtering, a gain curve and a One Euro filter; a double knock on the
> desk toggles it; a live dashboard shows every stage. The phone app delivers data in ~1 s
> batches, so the cursor runs smoothly but about 3 s behind the hand
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

## Move the cursor with the watch

```powershell
python scripts/pointer_live.py --note "pointer test" --subject S01 --wrist L
```

The pointer starts **off**. Knock your knuckles on the desk twice (or press F8) to toggle
it. Close the dashboard window or press Ctrl+C to stop; every run is recorded. Add
`--dry-run` to run everything without moving the real cursor.

Try the pointer without the watch, on a recorded session:

```powershell
python scripts/pointer_replay.py data/raw/sessions/<timestamp> --engaged
python scripts/pointer_replay.py data/raw/sessions/<timestamp> --live --dry-run
```

The first prints a summary and saves a cursor-path plot; the second replays the packets
with their original timing through the live loop and dashboard. Compare our orientation
filter with the watch's own:

```powershell
python scripts/compare_orientation.py --session S1=data/raw/sessions/<timestamp>
```

Settings live in `configs/dsp.yaml` and `configs/pointer.yaml`.

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
