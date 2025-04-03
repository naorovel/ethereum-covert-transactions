import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from generation import VGAE, VGAELoss
import numpy as np

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

# Training function with batch handling
def train_vgae(model, optimizer, loss_fn, dataset, epochs=200):
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # We pre-batched during preprocessing
        shuffle=True,
        collate_fn=collate_fn
    )
    
    writer = SummaryWriter()
    progress_bar = tqdm(total=epochs * len(dataloader), desc="Training Progress")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
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
            
            epoch_loss += loss.item()
        
        # Epoch logging
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/Epoch', avg_loss, epoch)

    progress_bar.close()
    writer.close()
    visualize_training()

# Using with transactions CSV
if __name__ == "__main__":
    # Initialize dataset and loader
    dataset = LargeTransactionDataset('src/data/transactions.csv', batch_size=2048)
    
    # Initialize the model
    model = VGAE(in_channels=1, hidden_dim=64, latent_dim=32)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Initialize loss function
    loss_fn = VGAELoss(beta=0.1)
    
    # Train with visualization
    train_vgae(model, optimizer, loss_fn, dataset)    
    
    # Access TensorBoard logs:
    # $ tensorboard --logdir=runs
    
    
    