import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import load_data, preprocess_data, split_baseline_stream
from src.autoencoder import Autoencoder


def train_model_on_array(
    data,
    epochs=20,
    batch_size=256,
    learning_rate=1e-3
):
    """
    Core training routine, decoupled from any file I/O. Trains a brand-new
    Autoencoder (sized to data's feature count) on an already-preprocessed
    numpy array and returns the trained model in memory — it is NOT saved
    to disk here.

    This is shared by:
      - train_autoencoder() below (the CLI entry point / `python -m src.train`),
        which preprocesses a CSV, trains on it, and saves autoencoder.pth.
      - src.detect_drift.train_and_detect_drift(), which trains a fresh model
        on just the uploaded reference dataset for a single API analysis run.

    Keeping the actual training loop in one place means a manual CLI train
    and an in-UI analysis run always train identically.
    """

    data_tensor = torch.tensor(
        data,
        dtype=torch.float32
    )

    dataset = TensorDataset(data_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    input_dim = data_tensor.shape[1]

    print(f"Training autoencoder with {input_dim} input features...")

    model = Autoencoder(input_dim)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for batch in dataloader:

            x = batch[0]

            optimizer.zero_grad()

            reconstructed = model(x)

            loss = criterion(
                reconstructed,
                x
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"- Reconstruction Loss: {avg_loss:.6f}"
        )

    model.eval()

    return model


def train_autoencoder(
    data_path,
    epochs=20,
    batch_size=256,
    learning_rate=1e-3
):
    """
    CLI / manual-training entry point. Loads a CSV from disk, preprocesses
    it, trains only on its baseline portion (via split_baseline_stream, as
    before), and saves the resulting weights to autoencoder.pth.

    This is now OPTIONAL for the API flow — POST /api/detect-drift trains a
    fresh model per-run from the uploaded reference dataset instead (see
    src.detect_drift.train_and_detect_drift). This function, and the
    autoencoder.pth it produces, remain available as a fallback/reference
    artifact and for offline experimentation.
    """

    # STEP 1: Load and preprocess data

    df = load_data(data_path)

    data, scaler = preprocess_data(df)

    # Train only on the baseline portion
    baseline_data, _ = split_baseline_stream(data)


    # STEP 2-4: Train the model on the baseline data

    model = train_model_on_array(
        baseline_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )


    # STEP 5: Save model

    project_root = Path(__file__).resolve().parent.parent

    model_path = project_root / "autoencoder.pth"

    torch.save(
        model.state_dict(),
        model_path
    )

    print(f"\nModel saved successfully:")
    print(model_path)

    return model, scaler


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_csv>")
        sys.exit(1)

    data_path = sys.argv[1]

    train_autoencoder(data_path)
