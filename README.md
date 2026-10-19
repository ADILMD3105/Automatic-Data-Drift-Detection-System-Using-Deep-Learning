# DriftDetect AI

An autoencoder-based dataset drift monitor. The frontend (`index.html`) is a
self-contained visual UI; `server.py` is a small Flask bridge that connects
it to the real detection pipeline in `src/` (`data_loader.py`,
`autoencoder.py`, `detect_drift.py`) and the trained weights
(`autoencoder.pth`).

## Project layout

```
DriftDetect/
├── index.html          # frontend (unchanged visual design, real backend calls)
├── server.py            # Flask bridge: serves index.html + POST /api/detect-drift
├── autoencoder.pth       # trained autoencoder weights (30 input features)
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data_loader.py    # load_data / preprocess_data / split_baseline_stream
    ├── autoencoder.py     # Autoencoder model definition
    ├── detect_drift.py    # detect_drift() — the real analysis entry point
    └── train.py           # trains a new autoencoder.pth from a baseline CSV
```

## Setup

```bash
cd DriftDetect
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python3 server.py
```

Then open **http://localhost:5000** in your browser. That's it — the
frontend is served by the same process that runs the detector, so there's
no separate dev server and no CORS setup needed.

By default it listens on port 5000. Override with `PORT=8000 python3
server.py` if that port is taken.

## Using it

1. Drop or select a **reference** (baseline) dataset and a **current**
   (stream) dataset into the two dropzones — `.csv`, `.txt` (semicolon
   -delimited), or `.xlsx`.
2. Click **Run Analysis**. The frontend sends both files to
   `POST /api/detect-drift`.
3. The server loads them with `src/data_loader.py`, runs them through the
   trained autoencoder in `src/detect_drift.py`, computes the 95th
   -percentile reconstruction-error threshold on the reference set, and
   flags current-set samples above it.
4. Real results (threshold, drift ratio, per-sample reconstruction errors,
   drift flags) are streamed back and rendered in the existing chart,
   metrics, and AI Core visualization — nothing is fabricated client-side.

## Important: the bundled model expects 30 numeric features

`autoencoder.pth` was trained with an input dimension of **30** (inspected
directly from the saved weight shapes). After `preprocess_data()` drops any
non-numeric first column and any all-missing/no-variance columns, your
reference **and** current datasets must both resolve to exactly 30 numeric
feature columns, or `detect_drift()` will raise a `ValueError` for
mismatched feature counts — which the UI will show as an analysis error
rather than a fake result.

If you want to use your own dataset shape, retrain first:

```bash
python3 -m src.train path/to/baseline.csv
```

This trains a new autoencoder on the baseline portion of that file (see
`split_baseline_stream` in `src/data_loader.py`, 60% baseline / 40% stream
by default) and overwrites `autoencoder.pth` in the project root.

## Missing pieces / things to verify on your machine

- **PyTorch could not be installed or exercised in this sandbox** (no
  network access here), so the tensor math itself was not run end-to-end
  in this environment. Everything *around* it was verified directly:
  the Flask routes, file upload handling, `data_loader.py`'s CSV/TXT/XLSX
  parsing, `detect_drift.py`'s control flow, and the exact JSON contract
  the frontend expects were all tested with a stand-in autoencoder. Once
  you `pip install -r requirements.txt` (which pulls real `torch`) on a
  machine with network access, the only remaining variable is the actual
  model weights — verify `autoencoder.pth` loads and expects 30 input
  features as noted above.
- No dataset was provided in this handoff, so drift results couldn't be
  verified against a real reference/current pair — you'll need to supply
  one that matches the 30-feature shape above (or retrain).
- CORS/auth are intentionally out of scope — this is a local single-user
  bridge server, not a hardened deployment.
