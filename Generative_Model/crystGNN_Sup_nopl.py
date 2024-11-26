import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

class CrystGNN_Supervise(nn.Module):
    """
    GNN model for fitting the supervised objectives for crystals.
    """

    def __init__(self, encoder, scaler, data_config):
        super().__init__()
        self.encoder = encoder  # Encoder module, instantiated outside
        self.scaler = scaler  # Scaler object for inverse transforms
        self.data_config = data_config  # Configuration for data properties

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        preds = self.encoder(batch)  # shape (N, 1)
        return preds

    def compute_loss_and_stats(self, batch, preds, prefix):
        loss = F.mse_loss(preds, batch.y)

        # Scaling predictions for metrics
        self.scaler.match_device(preds)
        scaled_preds = self.scaler.inverse_transform(preds)
        scaled_y = self.scaler.inverse_transform(batch.y)
        mae = torch.mean(torch.abs(scaled_preds - scaled_y))

        log_dict = {
            f'{prefix}_loss': loss.item(),
            f'{prefix}_mae': mae.item(),
        }

        if self.data_config['prop'] == 'scaled_lattice':
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]
            if self.data_config['lattice_scale_method'] == 'scale_length':
                pred_lengths *= batch.num_atoms.view(-1, 1).float() ** (1 / 3)
            
            lengths_mae = torch.mean(torch.abs(pred_lengths - batch.lengths))
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mard = mard(batch.angles, pred_angles)

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            log_dict.update({
                f'{prefix}_lengths_mae': lengths_mae.item(),
                f'{prefix}_angles_mae': angles_mae.item(),
                f'{prefix}_lengths_mard': lengths_mard.item(),
                f'{prefix}_angles_mard': angles_mard.item(),
                f'{prefix}_volumes_mard': volumes_mard.item(),
            })

        return log_dict, loss

# Example Training Loop
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs):
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}  # Move batch to device
            optimizer.zero_grad()
            preds = model(batch)
            loss = F.mse_loss(preds, batch['y'])  # Replace with loss function
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: val.to(device) for key, val in batch.items()}
                preds = model(batch)
                log_dict, loss = model.compute_loss_and_stats(batch, preds, prefix='val')
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")

    print("Training Complete.")

encoder = ...  # Define or load the encoder
scaler = ...  # Define or load the scaler
data_config = {'prop': 'scaled_lattice', 'lattice_scale_method': 'scale_length'}
model = CrystGNN_Supervise(encoder, scaler, data_config)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loader = ...
val_loader = ...

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(model, train_loader, val_loader, optimizer, device, num_epochs=50)
