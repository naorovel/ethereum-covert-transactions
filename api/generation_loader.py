import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from generation import VGAE, VGAELoss
import numpy as np
import os
from pathlib import Path

class LargeTransactionDataset(Dataset):
    def __init__(self, csv_path, batch_size=128):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.node_map = {}
        self.all_edges = []
        self.x = None
        self._preprocess()
        self.num_batches = len(self.all_edges) // self.batch_size

    def _preprocess(self):
        """Process entire graph at once with proper batching"""
        print("Preprocessing transaction data...")
        df = pd.read_csv(self.csv_path)
        
        # Create node mapping
        unique_nodes = pd.unique(df[['from_address', 'to_address']].values.ravel('K'))
        self.node_map = {node: idx for idx, node in enumerate(unique_nodes)}
        num_nodes = len(self.node_map)
        
        # Create full edge index array
        src = df['from_address'].map(self.node_map).to_numpy()
        tgt = df['to_address'].map(self.node_map).to_numpy()
        edges = np.stack([src, tgt], axis=0)
        
        # Split into batches during preprocessing
        self.all_edges = np.split(
            edges,
            np.arange(self.batch_size, edges.shape[1], self.batch_size),
            axis=1
        )
        
        # Create node features (degree-based)
        degrees = torch.zeros(num_nodes)
        for s, t in zip(src, tgt):
            degrees[s] += 1
            degrees[t] += 1
        self.x = degrees.unsqueeze(1).float()

    def __len__(self):
        return len(self.all_edges)

    def __getitem__(self, idx):
        """Return pre-batched edges"""
        return {
            'x': self.x,
            'edge_index': torch.tensor(self.all_edges[idx], dtype=torch.long)
        }

def collate_fn(batch):
    """Simplified collate for pre-batched data"""
    return batch[0]

def visualize_training(log_path='src/data/training_logs.csv'):
    
    try:
        # Create parent directories if needed
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(log_path):
            print(f"No training logs found at {log_path}")
            return

        """Plot training metrics"""
        df = pd.read_csv(log_path)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['loss'], label='Total Loss')
        plt.plot(df['epoch'], df['recon_loss'], label='Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['kl_div'], label='KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('KL')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()
    except Exception as e:
        print(f"Could not visualize training: {str(e)}")

# Training function with batch handling
# def train_vgae(model, optimizer, loss_fn, dataset, epochs=200):
#     dataloader = DataLoader(
#         dataset,
#         batch_size=1,  # We pre-batched during preprocessing
#         shuffle=True,
#         collate_fn=collate_fn
#     )
    
#     log_data = []
    
#     writer = SummaryWriter()
#     progress_bar = tqdm(total=epochs * len(dataloader), desc="Training Progress")
    
#     for epoch in range(epochs):
#         model.train()
#         epoch_loss = 0.0
        
#         for batch_idx, batch in enumerate(dataloader):
#             optimizer.zero_grad()
            
#             x = batch['x']  # Shape [num_nodes, 1]
#             edge_index = batch['edge_index']  # Shape [2, batch_size]
            
#             # Skip empty batches
#             if edge_index.shape[1] == 0:
#                 continue
                
#             # Forward pass
#             z, mu, logvar = model(x, edge_index)
            
#             # Calculate loss
#             loss, recon_loss, kl_loss = loss_fn(
#                 model, x, edge_index, edge_index
#             )
            
#             # Backpropagation
#             loss.backward()
#             optimizer.step()
            
#             # Update progress bar
#             progress_bar.update(1)
#             progress_bar.set_postfix({
#                 'epoch': epoch+1,
#                 'loss': f"{loss.item():.4f}",
#                 'recon': f"{recon_loss.item():.4f}",
#                 'kl': f"{kl_loss.item():.4f}"
#             })
            
#             epoch_loss += loss.item()
        
#         # Epoch logging
#         avg_loss = epoch_loss / len(dataloader)
#         avg_recon = recon_loss / len(dataloader)
#         avg_kl = kl_loss / len(dataloader)
        
#         log_data.append({
#             'epoch': epoch,
#             'loss': avg_loss,
#             'recon_loss': avg_recon,
#             'kl_div': avg_kl
#         })

#         writer.add_scalar('Loss/Epoch', avg_loss, epoch)

#     progress_bar.close()
#     writer.close()
    
#     # Save logs with directory creation
#     log_path = 'src/data/training_logs.csv'
#     try:
#         Path(log_path).parent.mkdir(parents=True, exist_ok=True)
#         pd.DataFrame(log_data).to_csv(log_path, index=False)
#     except Exception as e:
#         print(f"Could not save training logs: {str(e)}")
    
#     visualize_training(log_path)
    
def train_vgae(model, optimizer, loss_fn, dataset, epochs=200):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    log_data = []
    writer = SummaryWriter()
    
    try:
        progress_bar = tqdm(total=epochs * len(dataloader), desc="Training Progress")
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
            
                x = batch['x']  # Shape [num_nodes, 1]
                edge_index = batch['edge_index']  # Shape [2, batch_size]
                
                # Skip empty batches
                if edge_index.shape[1] == 0:
                    continue
                    
                # Forward pass
                z, mu, logvar = model(x, edge_index)
                
                # Calculate loss
                loss, recon_loss, kl_loss = loss_fn(
                    model, x, edge_index, edge_index
                )
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'epoch': epoch+1,
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{recon_loss.item():.4f}",
                    'kl': f"{kl_loss.item():.4f}"
                })
                
                # Accumulate losses
                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
            
            # Calculate epoch averages
            avg_loss = epoch_loss / len(dataloader)
            avg_recon = epoch_recon / len(dataloader)
            avg_kl = epoch_kl / len(dataloader)
            
            # Add to log data
            log_data.append({
                'epoch': epoch,
                'loss': avg_loss,
                'recon_loss': avg_recon,
                'kl_div': avg_kl
            })
            
        # Save logs after all epochs
        if log_data:
            log_path = 'src/data/training_logs.csv'
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(log_data).to_csv(log_path, index=False)
            
    finally:
        progress_bar.close()
        writer.close()
        visualize_training()

def merge_transactions(transactions, node_map, R=1000, F=10):
    """
    Merge transactions according to the pseudocode rules
    Args:
        transactions: List of (source, target) tuples
        node_map: Dictionary mapping node IDs to addresses
        R: Target number of transactions
        F: Allowed failures
    """
    # Convert indices back to addresses
    reverse_map = {v: k for k, v in node_map.items()}
    S = [{
        'input': {reverse_map[s]},
        'output': {reverse_map[t]}
    } for s, t in transactions]
    
    failures = 0
    while len(S) > R and failures < F:
        # Randomly select two distinct records
        idx_a, idx_b = np.random.choice(len(S), 2, replace=False)
        A = S[idx_a]
        B = S[idx_b]
        
        # Check for address overlap
        a_input = A['input']
        a_output = A['output']
        b_input = B['input']
        b_output = B['output']
        
        if not (a_input & b_output) and not (b_input & a_output):
            # Merge transactions
            C = {
                'input': a_input | b_input,
                'output': a_output | b_output
            }
            # Remove original transactions and add merged
            del S[max(idx_a, idx_b)]
            del S[min(idx_a, idx_b)]
            S.append(C)
            failures = 0  # Reset failure counter
        else:
            failures += 1
            
    return S[:R]  # Return up to R transactions

def save_transactions(transactions, filename):
    """Save merged transactions to CSV"""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    records = []
    for tx in transactions:
        records.append({
            'input_addresses': ';'.join(tx['input']),
            'output_addresses': ';'.join(tx['output'])
        })
        
    pd.DataFrame(records).to_csv(filename, index=False)


# Using with transactions CSV
if __name__ == "__main__":
    # Initialize dataset and loader
    dataset = LargeTransactionDataset('src/data/transactions.csv', batch_size=16384)
    
    # Initialize the model
    model = VGAE(in_channels=1, hidden_dim=64, latent_dim=32)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Initialize loss function
    loss_fn = VGAELoss(beta=0.1)
    
    # Train with visualization
    train_vgae(model, optimizer, loss_fn, dataset)
    
    # Generate and save covert transactions
    model.eval()
    with torch.no_grad():
        z, _, _ = model(dataset.x, dataset.all_edges[0])  # Use first batch's edges
        new_edge_probs = torch.sigmoid(z @ z.t())
        new_edges = (new_edge_probs > 0.5).nonzero(as_tuple=False).t()
    
    # Convert tensor to list of tuples
    edge_tuples = [(s.item(), t.item()) for s, t in new_edges.t()]
    
    # Merge and save transactions
    merged_transactions = merge_transactions(
        edge_tuples,
        dataset.node_map,
        R=1000,  # Target number of transactions
        F=20      # Allowed failures
    )
    save_transactions(merged_transactions, 'src/data/generated_transactions.csv')
    
    # Access TensorBoard logs:
    # $ tensorboard --logdir=runs
    
    
    