import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
import csv

class SparseGCNLayer(nn.Module):
    """GCN layer optimized for sparse adjacency matrices"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj_sparse):
        # adj_sparse: sparse tensor [n_nodes, n_nodes]
        # x: dense tensor [n_nodes, in_features]
        x = torch.sparse.mm(adj_sparse, x)  # Sparse-dense matrix multiplication
        return self.linear(x)

class SparseEncoder(nn.Module):
    """Encoder for sparse graph inputs"""
    def __init__(self, in_features, hidden_dim, latent_dim):
        super().__init__()
        self.gcn_mu = SparseGCNLayer(in_features, latent_dim)
        self.gcn_logvar = SparseGCNLayer(in_features, latent_dim)
        self.logvar_clamp_min = -10  # More conservative clamping for stability
        self.logvar_clamp_max = 3
    
    def forward(self, x, adj_sparse):
        mu = self.gcn_mu(x, adj_sparse)
        logvar = self.gcn_logvar(x, adj_sparse)
        return mu, torch.clamp(logvar, self.logvar_clamp_min, self.logvar_clamp_max)

class SparseDecoder(nn.Module):
    """Decoder with edge sampling for large graphs"""
    def forward(self, z, edge_index):
        # Calculate logits only for sampled edges
        rows, cols = edge_index
        z_rows = z[rows]
        z_cols = z[cols]
        return torch.sum(z_rows * z_cols, dim=1)  # Inner product for sampled edges

class SparseVGAE(nn.Module):
    """Sparse Variational Graph Autoencoder"""
    def __init__(self, in_features, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = SparseEncoder(in_features, hidden_dim, latent_dim)
        self.decoder = SparseDecoder()
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def forward(self, x, adj_sparse, edge_index):
        mu, logvar = self.encoder(x, adj_sparse)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, edge_index), mu, logvar

def sparse_loss(recon_logits, edge_label, mu, logvar, neg_samples=1):
    # Negative sampling loss implementation
    pos_loss = F.binary_cross_entropy_with_logits(
        recon_logits, edge_label, reduction='sum'
    )
    
    # KL divergence (same as before)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return pos_loss + kl_loss

def create_sparse_adjacency(df):
    # Create sparse adjacency matrix from transaction data
    all_nodes = pd.concat([df['from_address'], df['to_address']]).unique()
    node_map = {n: i for i, n in enumerate(all_nodes)}
    
    edge_list = []
    for _, row in df.iterrows():
        src = node_map[row['from_address']]
        dst = node_map[row['to_address']]
        edge_list.append([src, dst])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return edge_index, len(all_nodes)

def prepare_features(edge_index, n_nodes):
    # Create degree-based features
    in_degree = torch.zeros(n_nodes, dtype=torch.float32)
    out_degree = torch.zeros(n_nodes, dtype=torch.float32)
    
    src, dst = edge_index
    out_degree = torch.bincount(src, minlength=n_nodes).float()
    in_degree = torch.bincount(dst, minlength=n_nodes).float()
    
    features = torch.stack([in_degree, out_degree], dim=1)
    features = StandardScaler().fit_transform(features.numpy())
    return torch.tensor(features, dtype=torch.float32)  # Add this conversion
    
def edge_masking(edge_index, mask_rate=0.3):
    # Randomly mask edges while maintaining direction
    n_edges = edge_index.size(1)
    mask = torch.rand(n_edges) > mask_rate
    return edge_index[:, mask], edge_index[:, ~mask]

def embed_fake_transactions(model, features, edge_index, address_map,
                           fake_addresses, threshold=0.7, 
                           context_ratio=0.3, batch_size=5000,
                           max_new_ratio=0.1):  # Added 10% cap parameter
    """
    Embeds fake transactions (max 10% of original graph size)
    """
    fake_transactions = []
    n_real = features.size(0)
    device = features.device
    feat_dim = model.encoder.gcn_mu.linear.in_features
    
    # Calculate 10% of original transaction count
    original_edge_count = edge_index.size(1)
    max_new = max(1, int(original_edge_count * max_new_ratio))  # At least 1 transaction
    new_count = 0
    
    # 2. Generate fake features matching original feature space
    fake_features = torch.randn(len(fake_addresses), feat_dim, device=device) * 0.1
    fake_indices = torch.arange(n_real, n_real + len(fake_addresses), device=device)
    
    # 3. Create extended graph with isolated fake nodes
    combined_features = torch.cat([features, fake_features], dim=0)
    
    # Create extended adjacency (original edges + zero-padded for fake nodes)
    extended_adj = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.size(1), dtype=torch.float32, device=device),
        (combined_features.size(0), combined_features.size(0)))
    
    # 4. Generate combined embeddings
    with torch.no_grad():
        mu_combined, logvar_combined = model.encoder(combined_features, extended_adj)
        z_combined = model.reparameterize(mu_combined, logvar_combined)
    
    # 5. Calculate probabilities
    #adj_probs = torch.sigmoid(z_combined @ z_combined.t())
    
    # 6. Generate transactions
    # Fake -> Real connections
    for i in range(0, len(fake_indices), batch_size):
        if new_count >= max_new: break
        batch_fake = fake_indices[i:i+batch_size]
        #probs = adj_probs[batch_fake, :n_real]
        batch_z = z_combined[batch_fake,:]
        probs = torch.sigmoid(batch_z @ z_combined.t())[:, :n_real]
        top_real = select_context_nodes(probs, context_ratio, threshold)
        
        for fake_idx, real_indices in top_real:
            for real_idx in real_indices:
                if new_count >= max_new: break
                actual_fake_node = batch_fake[fake_idx].item()
                fake_transactions.append((
                    fake_addresses[actual_fake_node - n_real],
                    get_real_address(real_idx.item(), address_map),
                    probs[fake_idx, real_idx].item()
                ))
                new_count += 1
            else: continue
            break

    # Modified Real -> Fake connections
    if new_count < max_new:
        real_indices = torch.arange(n_real, device=device)
        for i in range(0, n_real, batch_size):
            if new_count >= max_new: break
            batch_real = real_indices[i:i+batch_size]
            #probs = adj_probs[batch_real, fake_indices]
            batch_z = z_combined[batch_real,:]
            probs = torch.sigmoid(batch_z @ z_combined.t())[:, :fake_indices]
            top_fake = select_context_nodes(probs, context_ratio, threshold)
            
            for real_idx, fake_idxs in top_fake:
                for fake_idx in fake_idxs:
                    if new_count >= max_new: break
                    actual_fake_node = fake_indices[fake_idx].item()
                    fake_transactions.append((
                        get_real_address(real_idx, address_map),
                        fake_addresses[actual_fake_node - n_real],
                        probs[real_idx, fake_idx].item()
                    ))
                    new_count += 1
                else: continue
                break

    # Modified Fake <-> Fake connections
    if new_count < max_new:
        #fake_probs = adj_probs[fake_indices, :][:, fake_indices]
        batch_z = z_combined[fake_indices,:]
        probs = torch.sigmoid(batch_z @ z_combined.t())[:, :fake_indices]
        fake_pairs = torch.nonzero(fake_probs > threshold)
        for src, dst in fake_pairs:
            if new_count >= max_new: break
            fake_transactions.append((
                fake_addresses[src.item()],
                fake_addresses[dst.item()],
                fake_probs[src, dst].item()
            ))
            new_count += 1
    
    return pd.DataFrame(fake_transactions,
                      columns=['input_address', 'output_address', 'probability'])

# Helper functions
def select_context_nodes(probs, context_ratio, threshold):
    """Select contextually relevant nodes using adaptive thresholding"""
    sorted_probs, indices = torch.sort(probs, descending=True)
    n_select = max(int(probs.size(1) * context_ratio), 1)
    mask = (sorted_probs > threshold) & (indices < sorted_probs.size(1))
    return [(idx, indices[idx][mask[idx]]) for idx in range(probs.size(0))]

def get_real_address(idx, address_map):
    return next(k for k, v in address_map.items() if v == idx)

    
def generate_from_input(model, features, adj_sparse, address_to_idx):
    """Generate transactions conditioned on input graph"""
    model.eval()
    with torch.no_grad():
        adj_logits, mu, logvar = model(features, adj_sparse)
        adj_probs = torch.sigmoid(adj_logits)
    return adj_probs

def batched_generation(z, batch_size=1000):
    probs = []
    for i in range(0, z.size(0), batch_size):
        batch = z[i:i+batch_size]
        probs.append(torch.sigmoid(batch @ z.t()))
    return torch.cat(probs)

def create_address_mapping(df):
    """Create and save address mapping from transaction data"""
    all_addresses = pd.concat([df['from_address'], df['to_address']]).unique()
    mapping = {addr: idx for idx, addr in enumerate(sorted(all_addresses))}
    
    # Save for reproducibility
    pd.Series(mapping).to_csv("src/data/address_mapping.csv")
    return mapping

# Example usage
if __name__ == "__main__":
    # Load and prepare data
        
    df = pd.read_csv("src/data/transactions.csv")
    
    original_edge_idx, n_nodes = create_sparse_adjacency(df)
    original_features = prepare_features(original_edge_idx, n_nodes)
    original_mapping = create_address_mapping(df)
    
    address_to_idx = create_address_mapping(df)
    
    edge_index, n_nodes = create_sparse_adjacency(df)
    features = prepare_features(edge_index, n_nodes)
    
    # Model parameters
    feat_dim = features.shape[1]
    hidden_dim = 16
    latent_dim = 8
    epochs = 200
    
    # Initialize model and optimizer
    model = SparseVGAE(feat_dim, hidden_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Edge masking
        visible_edges, masked_edges = edge_masking(edge_index, mask_rate=0.3)
        
        # Convert to sparse format for encoder
        adj_sparse = torch.sparse_coo_tensor(
            visible_edges, 
            torch.ones(visible_edges.size(1)),
            (n_nodes, n_nodes)
        )
        
        # Forward pass
        logits, mu, logvar = model(
            features, 
            adj_sparse,
            edge_index  # All edges for reconstruction
        )
        
        # Create labels: 1 for real edges, 0 for negative samples
        pos_labels = torch.ones(edge_index.size(1))
        
        # Calculate loss
        loss = sparse_loss(logits, pos_labels, mu, logvar)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.2f}")
            
    address_mapping = pd.read_csv("src/data/address_mapping.csv", index_col=0)
    address_to_idx = address_mapping['0'].to_dict()
    idx_to_address = {v: k for k, v in address_to_idx.items()}

    with open("src/data/fakeAddresses_unique.txt") as f:
        fake_addresses = [line.strip() for line in f]

    # Generate covert transactions
    covert_df = embed_fake_transactions(
        model=model,
        features=original_features,
        edge_index=original_edge_idx,
        address_map=address_to_idx,
        fake_addresses=fake_addresses,
        threshold=0.65,
        context_ratio=0.2,
        batch_size=1000,
        max_new_ratio=0.1
    )

    # Save results
    covert_df.to_csv("src/data/embedded_transactions.csv", index=False)
