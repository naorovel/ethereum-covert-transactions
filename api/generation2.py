import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, train_test_split_edges
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
import math


class VGAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(VGAE, self).__init__()
        # Encoder layers
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logvar = GCNConv(in_channels, out_channels)
        
        # Initialize weights properly
        nn.init.xavier_normal_(self.conv_mu.lin.weight)
        nn.init.xavier_normal_(self.conv_logvar.lin.weight)

    def encode(self, x, edge_index):
        mu = self.conv_mu(x, edge_index)
        mu = F.relu(mu)
        logvar = self.conv_logvar(x, edge_index)
        logvar = F.relu(logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, edge_index):
        src, dst = edge_index
        logits = (z[src] * z[dst]).sum(dim=1)
        return logits

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

def load_and_preprocess_data(real_transactions_path, fake_addresses_path):
    # Load data
    real_transactions = pd.read_csv(real_transactions_path)
    fake_addresses = pd.read_csv(fake_addresses_path, sep=" ")['address'].unique()

    # Create node mapping
    real_nodes = pd.unique(real_transactions[['from_address', 'to_address']].values.ravel('K'))
    all_nodes = np.concatenate([real_nodes, fake_addresses])
    node_to_idx = {node: np.uint32(idx) for idx, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)

    # Create edge index (keep dense for splitting)
    edge_pairs = real_transactions[['from_address', 'to_address']].values
    edges = [(node_to_idx[src], node_to_idx[dst]) for src, dst in edge_pairs]
    edge_index = torch.tensor(edges, dtype=torch.int32).t().contiguous()

    # Create node features
    in_degree = np.zeros(num_nodes, dtype=np.float32)
    out_degree = np.zeros(num_nodes, dtype=np.float32)
    for src, dst in edge_pairs:
        out_degree[node_to_idx[src]] += 1
        in_degree[node_to_idx[dst]] += 1

    # Normalize features
    X = np.stack([in_degree, out_degree], axis=1)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    X = torch.tensor(X, dtype=torch.float)

    # Create Data object (keep on CPU for splitting)
    data = Data(x=X, edge_index=edge_index)
    data.num_nodes = num_nodes

    # Convert to dense CPU tensor for splitting
    row, col = data.edge_index.cpu()


    # For directed graph
    unique_pairs = torch.unique(data.edge_index, dim=1)
    row, col = unique_pairs[0], unique_pairs[1]

    unique_edges = torch.unique(edge_index, dim=1)
    data.edge_index = unique_edges

    # Split edges
    val_ratio = 0.15
    test_ratio = 0.05
    n_v = int(val_ratio * row.size(0))
    n_t = int(test_ratio * row.size(0))
    
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    # Split indices
    test_idx = perm[:n_t]
    val_idx = perm[n_t:n_t + n_v]
    train_idx = perm[n_t + n_v:]

    # Store splits
    data.train_pos_edge_index = torch.stack([row[train_idx], col[train_idx]], dim=0)
    data.val_pos_edge_index = torch.stack([row[val_idx], col[val_idx]], dim=0)
    data.test_pos_edge_index = torch.stack([row[test_idx], col[test_idx]], dim=0)

    # Convert to sparse and move to device AFTER splitting
    data.edge_index = data.edge_index.to_sparse()
    data = data.to(device)

    # New validation checks for edge splits
    def _verify_split(edges, original_edges):
        edge_set = set(zip(edges[0].cpu().numpy(), edges[1].cpu().numpy()))
        original_set = set(zip(original_edges[0].cpu().numpy(), original_edges[1].cpu().numpy()))
        return len(edge_set - original_set) == 0  # No new edges introduced

    # Verify no overlap between splits
    train_set = set(zip(data.train_pos_edge_index[0].tolist(), data.train_pos_edge_index[1].tolist()))
    val_set = set(zip(data.val_pos_edge_index[0].tolist(), data.val_pos_edge_index[1].tolist()))
    test_set = set(zip(data.test_pos_edge_index[0].tolist(), data.test_pos_edge_index[1].tolist()))

    assert len(train_set & val_set) == 0, "Train-Val overlap detected"
    assert len(train_set & test_set) == 0, "Train-Test overlap detected"
    assert len(val_set & test_set) == 0, "Val-Test overlap detected"

    # Verify all edges are from original set
    original_edges = set(zip(row.tolist(), col.tolist()))
    assert train_set.issubset(original_edges), "Train contains new edges"
    assert val_set.issubset(original_edges), "Val contains new edges"
    assert test_set.issubset(original_edges), "Test contains new edges"

    # Verify coverage
    combined = train_set | val_set | test_set
    assert len(combined) == len(original_edges), f"Missing {len(original_edges)-len(combined)} edges"

    return data, node_to_idx, fake_addresses


def train_model(data, device, num_epochs=500, lr=0.001):
    model = VGAE(in_channels=data.x.size(1), hidden_dim=32, out_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Training edges
        train_edges = data.train_pos_edge_index
        neg_edge_index = negative_sampling(
            edge_index=train_edges,
            num_nodes=data.num_nodes,
            num_neg_samples=train_edges.size(1))

        z, mu, logvar = model(data.x, train_edges)  # Use training edges
        
        # Calculate loss for positive and negative edges
        pos_logits = model.decode(z, data.train_pos_edge_index.to(device))
        neg_logits = model.decode(z, neg_edge_index.to(device))
        
        pos_labels = torch.ones(pos_logits.size(0), device=device)
        neg_labels = torch.zeros(neg_logits.size(0), device=device)
        
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        # Calculate losses
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.num_nodes
        loss = bce_loss + kl_loss

        # Gradient handling
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=2.0)  # Gradient clipping
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            # Use training edges for encoding
            z_val, mu_val, logvar_val = model(data.x, train_edges)
            
            # Positive edges: data.val_pos_edge_index
            pos_logits = model.decode(z_val, data.val_pos_edge_index)
            
            # Sample negative edges for validation (not in training or validation)
            val_neg_edges = negative_sampling(
                edge_index=torch.cat([train_edges, data.val_pos_edge_index], dim=1),
                num_nodes=data.num_nodes,
                num_neg_samples=data.val_pos_edge_index.size(1))
            
            neg_logits = model.decode(z_val, val_neg_edges)
            
            # Combine and compute loss
            val_logits = torch.cat([pos_logits, neg_logits])
            val_labels = torch.cat([
                torch.ones_like(pos_logits),
                torch.zeros_like(neg_logits)
            ])
            val_loss = F.binary_cross_entropy_with_logits(val_logits, val_labels)

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return model

def generate_fake_transactions(model, data, node_to_idx, fake_addresses, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        z, _, _ = model(data.x.to(device), data.train_pos_edge_index.to(device))
    
    fake_indices = [node_to_idx[addr] for addr in fake_addresses]
    real_indices = list(set(range(data.num_nodes)) - set(fake_indices))
    
    # Generate potential fake transactions
    fake_edges = []
    real_edge_set = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    
    for fake_idx in fake_indices:
        # Generate outgoing edges
        logits_out = (z[fake_idx] * z[real_indices]).sum(dim=1)
        probs_out = torch.sigmoid(logits_out)
        mask_out = probs_out > threshold
        for i, idx in enumerate(real_indices):
            if mask_out[i] and (fake_idx, idx) not in real_edge_set:
                fake_edges.append((fake_idx, idx))
        
        # Generate incoming edges
        logits_in = (z[real_indices] * z[fake_idx]).sum(dim=1)
        probs_in = torch.sigmoid(logits_in)
        mask_in = probs_in > threshold
        for i, idx in enumerate(real_indices):
            if mask_in[i] and (idx, fake_idx) not in real_edge_set:
                fake_edges.append((idx, fake_idx))
    
    # Convert indices to addresses
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    fake_transactions = []
    for src_idx, dst_idx in fake_edges:
        src = idx_to_node[src_idx.item()]
        dst = idx_to_node[dst_idx.item()]
        fake_transactions.append({'from_address': src, 'to_address': dst})
    
    return pd.DataFrame(fake_transactions)

# Example usage
if __name__ == "__main__":
    print(torch.cuda.current_device())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    data, node_to_idx, fake_addresses = load_and_preprocess_data(
        'src/data/transactions.csv',
        'src/data/fakeAddresses_unique.txt'
    )
    
    # Train model
    trained_model = train_model(data, device)
    
    # Generate fake transactions
    fake_df = generate_fake_transactions(
        trained_model,
        data,
        node_to_idx,
        fake_addresses,
        device,
        threshold=0.7  # Adjust based on validation results
    )
    
    # Save results
    fake_df.to_csv('src/data/generated_transactions.csv', index=False)