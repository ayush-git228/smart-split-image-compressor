import torch
import torch.nn as nn
import numpy as np

class TinyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 32*32),
            nn.Sigmoid(),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def compress_tile_ae(tile, model=None):
    # Compresses a 32x32 grayscale tile via a simple pretrained autoencoder. Returns uint8 image
    # Normalize and tensorize
    
    device = torch.device("cpu")
    tens = torch.from_numpy(tile / 255.0).float().unsqueeze(0).to(device)
    if model is None:
        model = TinyAutoencoder()
        # For demo, use random weights (optionally: load a pre-trained model)
    model.eval()
    with torch.no_grad():
        recon = model(tens)
    out_img = recon.squeeze(0).detach().cpu().numpy().reshape(tile.shape)
    return np.clip(out_img * 255, 0, 255).astype(np.uint8)