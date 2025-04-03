import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, negative_sampling

class VGAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super(VGAE, self).__init__()
        # Encoder
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)
        
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z, edge_index):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)

    def forward(self, x, edge_index):
        # Ensure input dimensions
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Add feature dimension
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class VGAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super(VGAELoss, self).__init__()
        self.beta = beta
        
    def forward(self, model, x, pos_edge_index, all_edge_index):
        z, mu, logvar = model(x, pos_edge_index)
        
        # Positive edge reconstruction
        pos_pred = model.decode(z, pos_edge_index)
        
        # Negative edge sampling
        neg_edge_index = negative_sampling(
            edge_index=all_edge_index,
            num_nodes=x.size(0),
            num_neg_samples=pos_edge_index.size(1)
        )
        neg_pred = model.decode(z, neg_edge_index)
        
        # BCE reconstruction loss
        pos_loss = -torch.log(pos_pred + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()
        recon_loss = pos_loss + neg_loss
        
        # KL divergence
        kl_div = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        
        return recon_loss + self.beta * kl_div, recon_loss, kl_div

class EdgeMasker:
    def mask_edges(self, edge_index, num_nodes):
        # Create valid PyG Data object
        data = Data(
            x=torch.ones(num_nodes, 1),  # Dummy features required
            edge_index=edge_index,
            num_nodes=num_nodes
        )
        
        # Split edges - returns modified Data object with split attributes
        return train_test_split_edges(
            data,
            val_ratio=0.15,
            test_ratio=0.1
        )
