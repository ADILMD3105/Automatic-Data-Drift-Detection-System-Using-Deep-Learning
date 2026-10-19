import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # encoder: reduces input into smaller representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)   # bottleneck layer
        )

        # decoder: reconstructs data back to original size
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        # pass input through encoder
        latent = self.encoder(x)

        # reconstruct from compressed representation
        reconstructed = self.decoder(latent)

        return reconstructed