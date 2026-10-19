"""
server.py — DriftDetect AI bridge server
==========================================
The SMALLEST reliable bridge between the existing frontend
(index.html) and the existing Python detection pipeline
(src/data_loader.py, src/autoencoder.py, src/detect_drift.py).

It does not reimplement or alter any ML logic. It:
  1. Serves the existing index.html unchanged at "/".
  2. Accepts the reference/current file uploads from the
     frontend's dropzones at POST /api/detect-drift.
  3. Saves them to a temp dir (preserving their extension, since
     src/data_loader.load_data() dispatches on file extension).
  4. Loads them into DataFrames with load_data(), exactly as
     train.py / detect_drift.py already do.
  5. Calls the real train_and_detect_drift() from src/detect_drift.py,
     which trains a brand-new autoencoder on the uploaded reference
     dataset only, then scores both datasets against it.
  6. Returns JSON matching the exact contract index.html already
     expects:
       { threshold, drift_ratio, drift_flags[], reference_errors[],
         current_errors[], total_samples, drifted_samples,
         run_time_seconds }

No results are ever fabricated here. If preprocessing, loading,
or inference fails, a JSON error (with a real message) is
returned with an HTTP error status, and the frontend surfaces it
verbatim instead of showing fake numbers.
"""

import os
import tempfile
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src.data_loader import load_data
from src.detect_drift import train_and_detect_drift

PROJECT_ROOT = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {".csv", ".txt", ".xlsx"}

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(PROJECT_ROOT, "index.html")


def _save_upload(file_storage, tmp_dir):
    """Save an uploaded werkzeug FileStorage to disk, keeping its
    original extension so load_data() can dispatch correctly."""
    filename = file_storage.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f'Unsupported file type "{ext or "(none)"}" for "{filename}". '
            f"Supported formats: .csv, .txt, .xlsx"
        )
    dest = Path(tmp_dir) / f"upload{ext}"
    file_storage.save(dest)
    return dest


@app.route("/api/detect-drift", methods=["POST"])
def api_detect_drift():
    if "reference" not in request.files or "current" not in request.files:
        return jsonify({"error": 'Both "reference" and "current" files are required.'}), 400

    reference_file = request.files["reference"]
    current_file = request.files["current"]

    start_ts = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            reference_path = _save_upload(reference_file, tmp_dir)
            current_path = _save_upload(current_file, tmp_dir)

            reference_df = load_data(str(reference_path))
            current_df = load_data(str(current_path))

            result = train_and_detect_drift(reference_df, current_df)

    except ValueError as ve:
        # Expected, user-facing errors: bad format, mismatched feature
        # columns, no valid numeric features left, etc.
        return jsonify({"error": str(ve)}), 400
    except Exception:
        # Unexpected failure — surface it rather than silently mocking a
        # result. Full traceback goes to the server console for debugging.
        traceback.print_exc()
        return jsonify({"error": "The detector failed while analyzing the uploaded data. See server logs for details."}), 500

    run_time_seconds = time.time() - start_ts

    drift_flags = result["drift_flags"]
    current_errors = result["current_errors"]
    reference_errors = result["reference_errors"]

    response = {
        "threshold": float(result["threshold"]),
        "drift_ratio": float(result["drift_ratio"]),
        "drift_flags": [bool(f) for f in drift_flags],
        "reference_errors": [float(x) for x in reference_errors],
        "current_errors": [float(x) for x in current_errors],
        "total_samples": int(len(current_errors)),
        "drifted_samples": int(sum(bool(f) for f in drift_flags)),
        "run_time_seconds": run_time_seconds,
    }

    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"DriftDetect AI bridge server running at http://localhost:{port}")
    print("Each analysis trains a fresh autoencoder on the uploaded reference dataset.")
    app.run(host="0.0.0.0", port=port, debug=False)
