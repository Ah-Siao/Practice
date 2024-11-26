import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class CDVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, hidden_dim, 
                 fc_num_layers, max_atoms, sigma_params, type_sigma_params,
                 predict_property=False, cost_params=None, lattice_scaler=None, scaler=None):
        super(CDVAE, self).__init__()

        # Initialize encoder, decoder, and latent layers
        self.encoder = encoder
        self.decoder = decoder
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

        # MLPs for stats prediction
        self.fc_num_atoms = build_mlp(latent_dim, hidden_dim, fc_num_layers, max_atoms + 1)
        self.fc_lattice = build_mlp(latent_dim, hidden_dim, fc_num_layers, 6)
        self.fc_composition = build_mlp(latent_dim, hidden_dim, fc_num_layers, MAX_ATOMIC_NUM)

        # Optional property predictor
        self.predict_property = predict_property
        if self.predict_property:
            self.fc_property = build_mlp(latent_dim, hidden_dim, fc_num_layers, 1)

        # Initialize sigma values
        sigmas = torch.tensor(
            torch.exp(torch.linspace(torch.log(sigma_params[0]), torch.log(sigma_params[1]), sigma_params[2])),
            dtype=torch.float32
        )
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(
            torch.exp(torch.linspace(torch.log(type_sigma_params[0]), torch.log(type_sigma_params[1]), type_sigma_params[2])),
            dtype=torch.float32
        )
        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        # Predefined embeddings
        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        self.lattice_scaler = lattice_scaler
        self.scaler = scaler
        self.cost_params = cost_params or {}

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        hidden = self.encoder(batch)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def decode_stats(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None, teacher_forcing=False):
        # Decodes z to predict number of atoms, lattice stats, and compositions
        if gt_num_atoms is not None:
            num_atoms = self.fc_num_atoms(z)
            lengths_and_angles, lengths, angles = self.predict_lattice(z, gt_num_atoms)
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.fc_num_atoms(z).argmax(dim=-1)
            lengths_and_angles, lengths, angles = self.predict_lattice(z, num_atoms)
            composition_per_atom = self.predict_composition(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    def forward(self, batch, teacher_forcing=False):
        mu, log_var, z = self.encode(batch)
        decoded_stats = self.decode_stats(z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)
        return z, mu, log_var, decoded_stats

    def training_step(self, batch, optimizer):
        # Perform a single training step
        self.train()
        optimizer.zero_grad()
        z, mu, log_var, decoded_stats = self(batch, teacher_forcing=True)

        # Compute losses
        losses = self.compute_losses(batch, decoded_stats, mu, log_var)
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()

        return losses

    def validation_step(self, batch):
        # Perform a validation step
        self.eval()
        with torch.no_grad():
            z, mu, log_var, decoded_stats = self(batch, teacher_forcing=False)
            losses = self.compute_losses(batch, decoded_stats, mu, log_var)
        return losses

    def compute_losses(self, batch, decoded_stats, mu, log_var):
        pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles, pred_composition_per_atom = decoded_stats

        # Loss components
        num_atom_loss = F.cross_entropy(pred_num_atoms, batch.num_atoms)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(pred_composition_per_atom, batch.atom_types, batch)
        kld_loss = self.kld_loss(mu, log_var)

        losses = {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'kld_loss': kld_loss,
        }

        if self.predict_property:
            property_loss = self.property_loss(mu, batch)
            losses['property_loss'] = property_loss

        return losses

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        scaled_target = self.lattice_scaler.transform(
            torch.cat([batch.lengths, batch.angles], dim=-1)
        )
        return F.mse_loss(pred_lengths_and_angles, scaled_target)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom, target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)
