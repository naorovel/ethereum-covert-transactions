
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
        self.out_channels = out_channels  # Add this line
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, out_channels)
        self.conv_logvar = GCNConv(hidden_dim, out_channels)
        
        # Initialize weights properly
        nn.init.normal_(self.conv_mu.lin.weight, std=0.01)
        nn.init.normal_(self.conv_logvar.lin.weight, std=0.01)

    def encode(self, x, edge_index):
        # mu = self.conv_mu(x, edge_index)
        # # mu = F.relu(mu)
        # logvar = self.conv_logvar(x, edge_index)
        # logvar = F.relu(logvar)
        x = F.relu(self.conv1(x, edge_index))  # Activation only here
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)
        # return mu, logvar

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
        return logits/math.sqrt(self.out_channels)

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
    # data.train_pos_edge_index = torch.stack([row[train_idx], col[train_idx]], dim=0)
    # data.val_pos_edge_index = torch.stack([row[val_idx], col[val_idx]], dim=0)
    # data.test_pos_edge_index = torch.stack([row[test_idx], col[test_idx]], dim=0)

    # Convert to sparse and move to device AFTER splitting
    # data.edge_index = data.edge_index.to_sparse()
    # data = data.to(device)
    
    data.train_pos_edge_index = torch.stack([row[train_idx], col[train_idx]], dim=0)
    data.val_pos_edge_index = torch.stack([row[val_idx], col[val_idx]], dim=0)
    data.test_pos_edge_index = torch.stack([row[test_idx], col[test_idx]], dim=0)
    
    data.full_edge_index = data.edge_index.to_sparse().to(device)
    data.edge_index = data.train_pos_edge_index  # Use dense edges for training

    # Move ALL tensors to device after splitting
    data = data.to(device)
    data.train_pos_edge_index = data.train_pos_edge_index.to(device)
    data.val_pos_edge_index = data.val_pos_edge_index.to(device)
    data.test_pos_edge_index = data.test_pos_edge_index.to(device)
    data.full_edge_index = data.full_edge_index.to(device)

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


def train_model(data, device, num_epochs=200, lr=0.0005):

    model = VGAE(in_channels=data.x.size(1), hidden_dim=32, out_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Data x device: {data.x.device}")
    print(f"Train edges device: {data.train_pos_edge_index.device}")
    
    x = data.x.to(device)
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Training edges
        train_edges = data.train_pos_edge_index.to(device)
        neg_edge_index = negative_sampling(
            edge_index=torch.cat([train_edges, data.val_pos_edge_index], dim=1),
            num_nodes=data.num_nodes,
            num_neg_samples=train_edges.size(1) * 2).to(device)

        z, mu, logvar = model(x, train_edges)  # Use training edges
        
        # Calculate loss for positive and negative edges
        pos_logits = model.decode(z, data.train_pos_edge_index.to(device))
        neg_logits = model.decode(z, neg_edge_index.to(device))
        
        pos_labels = torch.ones(pos_logits.size(0), device=device)
        neg_labels = torch.zeros(neg_logits.size(0), device=device)
        
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        # Calculate losses
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        out_channels = 16
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (data.num_nodes * out_channels)
        loss = bce_loss + kl_loss

        # Gradient handling
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        print(f"Gradient norms: {np.mean(grad_norms):.4f}")
        clip_grad_norm_(model.parameters(), max_norm=2.0)  # Gradient clipping
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            # Use training edges for encoding
            z_val, mu_val, logvar_val = model(x, train_edges)
            
            # Positive edges: data.val_pos_edge_index
            val_edges = data.val_pos_edge_index.to(device)
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
            scheduler.step(val_loss)


        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return model

def generate_fake_transactions(model, data, node_to_idx, fake_addresses, device, threshold=0.95):
    # Convert edge_index to appropriate format (already dense tensor)
    edge_index = data.edge_index.cpu()
    real_edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))  # Directly use dense format
    
    model.eval()
    with torch.no_grad():
        z, _, _ = model(data.x.to(device), data.train_pos_edge_index.to(device))
    
    fake_indices = [node_to_idx[addr] for addr in fake_addresses]
    real_indices = list(set(range(data.num_nodes)) - set(fake_indices))
    
    # Generate potential fake transactions
    fake_edges = []
    
    # Process in chunks to avoid memory issues
    chunk_size = 1000
    for i in range(0, len(fake_indices), chunk_size):
        print(f"Processing chunk {i // chunk_size + 1}/{math.ceil(len(fake_indices) / chunk_size)}")
        
        chunk = fake_indices[i:i+chunk_size]
        
        # Convert indices to tensors on correct device
        fake_tensor = torch.tensor(chunk, device=device, dtype=torch.long)
        real_tensor = torch.tensor(real_indices, device=device, dtype=torch.long)
        
        # Calculate outgoing probabilities
        z_fake = z[fake_tensor]
        z_real = z[real_tensor]
        logits_out = torch.mm(z_fake, z_real.t())
        probs_out = torch.sigmoid(logits_out)
        
        # Find valid outgoing connections
        mask_out = probs_out > threshold
        src_ids, dst_ids = torch.where(mask_out)
        
        for src, dst in zip(src_ids, dst_ids):
            fake_idx = chunk[src.item()]
            real_idx = real_indices[dst.item()]
            if (fake_idx, real_idx) not in real_edge_set:
                fake_edges.append((fake_idx, real_idx))
        
        # Calculate incoming probabilities
        logits_in = torch.mm(z_real, z_fake.t())
        probs_in = torch.sigmoid(logits_in)
        
        # Find valid incoming connections
        mask_in = probs_in > threshold
        src_ids, dst_ids = torch.where(mask_in)
        
        for src, dst in zip(src_ids, dst_ids):
            real_idx = real_indices[src.item()]
            fake_idx = chunk[dst.item()]
            if (real_idx, fake_idx) not in real_edge_set:
                fake_edges.append((real_idx, fake_idx))
    
    # Convert indices to addresses
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    fake_transactions = []
    for src_idx, dst_idx in fake_edges:
        src = idx_to_node[src_idx]
        dst = idx_to_node[dst_idx]
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

    torch.cuda.empty_cache()
    
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