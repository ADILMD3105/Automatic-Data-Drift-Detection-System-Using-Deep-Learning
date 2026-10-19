import torch
import numpy as np

from src.data_loader import load_data, preprocess_data
from src.autoencoder import Autoencoder
from src.train import train_model_on_array


def compute_reconstruction_error(model, data):
    model.eval()

    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32)

        reconstructed = model(data_tensor)

        errors = torch.mean(
            (reconstructed - data_tensor) ** 2,
            dim=1
        )

    return errors.numpy()


def _drift_result_from_errors(reference_errors, current_errors):
    # Use reference data to calculate the drift threshold
    threshold = np.percentile(
        reference_errors,
        95
    )

    # Detect drift in current data
    drift_flags = current_errors > threshold

    drift_ratio = drift_flags.mean()

    return {
        "threshold": threshold,
        "drift_ratio": drift_ratio,
        "drift_flags": drift_flags,
        "reference_errors": reference_errors,
        "current_errors": current_errors,
    }


def train_and_detect_drift(
    reference_df,
    current_df,
    epochs=20,
    batch_size=256,
    learning_rate=1e-3
):
    """
    Full per-run drift-detection flow: trains a BRAND-NEW autoencoder using
    ONLY the uploaded reference dataset, then scores both the reference and
    current datasets against it. This is what the API server calls for every
    analysis request, so there's no separate manual-training step required
    before an upload.

      reference_df -> baseline/training distribution. Preprocessed, fit a
                       scaler on it, and train a fresh Autoencoder on the
                       WHOLE reference dataset (never on current_df).
      current_df   -> distribution being tested for drift. Preprocessed with
                       the SAME fitted scaler, then only ever run through the
                       trained model for inference — never used in training.

    The drift methodology itself (P95-of-reference-errors threshold,
    current-errors-over-threshold flags) is unchanged from detect_drift()
    below, and the return shape is identical.
    """

    # Preprocess reference data and fit the scaler on it only
    reference_data, scaler = preprocess_data(reference_df)

    # Preprocess current data using that SAME fitted scaler — current data
    # never influences fitting/training, only inference.
    current_data, _ = preprocess_data(
        current_df,
        scaler
    )

    # Make sure both datasets have the same number of features
    if reference_data.shape[1] != current_data.shape[1]:
        raise ValueError(
            "Reference and current datasets must have the same number of features."
        )

    # Train a fresh autoencoder for this run, sized automatically to the
    # reference dataset's feature count, using ONLY reference data.
    model = train_model_on_array(
        reference_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # Calculate reconstruction errors with the freshly trained model
    reference_errors = compute_reconstruction_error(
        model,
        reference_data
    )

    current_errors = compute_reconstruction_error(
        model,
        current_data
    )

    print(f"S {reference_errors.mean():.6f}")
    print(f"[DIAGNOSTIC] Current mean error: {current_errors.mean():.6f}")
    print(f"[DIAGNOSTIC] Reference P95 threshold: {np.percentile(reference_errors, 95):.6f}")
    print(f"[DIAGNOSTIC] Current P95 error: {np.percentile(current_errors, 95):.6f}")
    print(f"[DIAGNOSTIC] Max error difference: {np.max(np.abs(current_errors - reference_errors)):.6f}")
    print(f"[DIAGNOSTIC] Mean absolute error difference: {np.mean(np.abs(current_errors - reference_errors)):.6f}")
    print(f"[DIAGNOSTIC] Current errors identical to reference: {np.array_equal(current_errors, reference_errors)}")

    return _drift_result_from_errors(reference_errors, current_errors)


def detect_drift(reference_df, current_df, model_path="autoencoder.pth"):
    """
    Fallback/reference path: scores reference and current data against an
    already-trained checkpoint on disk, instead of training a fresh model
    for this run. Kept for offline experimentation and as a reference
    artifact — the live API flow uses train_and_detect_drift() above.
    """
    # Preprocess reference data and fit the scaler
    reference_data, scaler = preprocess_data(reference_df)

    # Preprocess current data using the same scaler
    current_data, _ = preprocess_data(
        current_df,
        scaler
    )

    # Make sure both datasets have the same number of features
    if reference_data.shape[1] != current_data.shape[1]:
        raise ValueError(
            "Reference and current datasets must have the same number of features."
        )

    # Load the trained checkpoint first, and build the autoencoder using the
    # input dimension it was actually trained with — not the column count of
    # whatever dataset happens to be uploaded. The architecture and weights
    # are unchanged; this only fixes *what shape we instantiate before
    # loading them*, so the checkpoint always loads cleanly.
    checkpoint = torch.load(model_path, map_location="cpu")
    trained_input_dim = checkpoint["encoder.0.weight"].shape[1]

    observed_input_dim = reference_data.shape[1]

    # If the uploaded dataset doesn't resolve (after the same preprocessing
    # used at training time) to the exact feature count the model was
    # trained on, that's a genuine schema mismatch — surface it clearly
    # instead of letting a raw state_dict size-mismatch crash the request.
    if observed_input_dim != trained_input_dim:
        raise ValueError(
            f"Preprocessed data has {observed_input_dim} feature column(s), but the "
            f"trained autoencoder at '{model_path}' expects {trained_input_dim}. "
            "The reference and current datasets must resolve, after preprocessing "
            "(dropping the non-numeric first column, low-variance/no-variance "
            "columns, and columns with too many missing values), to the same "
            "feature schema the model was trained on. Re-train the model on this "
            "dataset if you intend to use a different schema."
        )

    model = Autoencoder(trained_input_dim)
    model.load_state_dict(checkpoint)
    model.eval()

    # Calculate reconstruction errors
    reference_errors = compute_reconstruction_error(
        model,
        reference_data
    )

    current_errors = compute_reconstruction_error(
        model,
        current_data
    )

    return _drift_result_from_errors(reference_errors, current_errors)