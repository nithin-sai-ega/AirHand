# AirHand

Control a Windows laptop with an Apple Watch. Wrist rotation moves the cursor, and a
streaming neural network recognises discrete gestures (flick, twist, double tap, circle,
shake) to trigger actions. Everything runs on the laptop CPU, and model size, INT8
quantization and latency are measured as part of the project.

> Status: **Phase 0, the feasibility probe.** The repository can receive, log and analyse the
> real watch stream. Measured: 99.9 Hz with no gaps or loss, delivered in ~1 s batches
> (see [docs/RESULTS.md](docs/RESULTS.md#phase-0)).

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
