"""
POYO Simple Example - A minimal example demonstrating the POYO architecture with fake data.

This script creates a fake neural recording dataset and trains a simplified POYO model.
It's useful for understanding the data flow and model architecture without needing real data.

Usage:
    python train_simple.py
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Fake Dataset
# ============================================================================

@dataclass
class FakeRecordingConfig:
    """Configuration for generating fake neural recordings."""
    num_units: int = 50           # Number of neurons
    num_samples: int = 1000       # Number of training samples
    sequence_length: float = 1.0  # Duration in seconds
    firing_rate: float = 10.0     # Average spikes per second per neuron
    output_dim: int = 2           # Output dimension (e.g., 2D velocity)
    

class FakeNeuralDataset(Dataset):
    """
    A fake dataset that generates synthetic neural spike data and behavior labels.
    
    This mimics the structure expected by POYO:
    - Input: spike times and unit indices
    - Output: behavior predictions (e.g., velocity)
    """
    
    def __init__(
        self, 
        config: FakeRecordingConfig,
        split: str = "train",
        seed: int = 42
    ):
        self.config = config
        self.split = split
        self.rng = np.random.RandomState(seed + hash(split) % 1000)
        
        # Generate fake data
        self._generate_data()
        
    def _generate_data(self):
        """Generate fake spike trains and behavior targets."""
        self.samples = []
        
        for i in range(self.config.num_samples):
            # Generate random spike times for each unit
            spike_times = []
            spike_units = []
            
            for unit_idx in range(self.config.num_units):
                # Poisson process: number of spikes
                n_spikes = self.rng.poisson(
                    self.config.firing_rate * self.config.sequence_length
                )
                # Random spike times within the window
                times = self.rng.uniform(0, self.config.sequence_length, n_spikes)
                times = np.sort(times)
                
                spike_times.extend(times)
                spike_units.extend([unit_idx] * n_spikes)
            
            # Sort by time
            if len(spike_times) > 0:
                sort_idx = np.argsort(spike_times)
                spike_times = np.array(spike_times)[sort_idx]
                spike_units = np.array(spike_units)[sort_idx]
            else:
                spike_times = np.array([0.5])  # dummy spike
                spike_units = np.array([0])
            
            # Generate fake behavior target (random 2D velocity correlated with spike count)
            total_spikes = len(spike_times)
            # Make target somewhat correlated with neural activity
            target = self.rng.randn(self.config.output_dim) * 0.5
            target[0] += (total_spikes - self.config.num_units * self.config.firing_rate) * 0.01
            
            self.samples.append({
                "spike_times": spike_times.astype(np.float32),
                "spike_units": spike_units.astype(np.int64),
                "target": target.astype(np.float32),
                "session_id": f"session_{i % 5}",  # 5 fake sessions
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_unit_ids(self) -> List[str]:
        """Return list of unique unit IDs."""
        return [f"unit_{i}" for i in range(self.config.num_units)]
    
    def get_session_ids(self) -> List[str]:
        """Return list of unique session IDs."""
        return [f"session_{i}" for i in range(5)]


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function that pads spike sequences to the same length.
    
    Similar to torch_brain's collate but simplified.
    """
    # Find max sequence length
    max_len = max(len(s["spike_times"]) for s in batch)
    # Pad to multiple of 8 for efficiency
    max_len = ((max_len + 7) // 8) * 8
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    spike_times = torch.zeros(batch_size, max_len, dtype=torch.float32)
    spike_units = torch.zeros(batch_size, max_len, dtype=torch.int64)
    input_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    targets = torch.stack([torch.tensor(s["target"]) for s in batch])
    
    for i, sample in enumerate(batch):
        seq_len = len(sample["spike_times"])
        spike_times[i, :seq_len] = torch.tensor(sample["spike_times"])
        spike_units[i, :seq_len] = torch.tensor(sample["spike_units"])
        input_mask[i, :seq_len] = True
    
    return {
        "spike_times": spike_times,
        "spike_units": spike_units,
        "input_mask": input_mask,
        "targets": targets,
    }


# ============================================================================
# Simplified POYO Model
# ============================================================================

class RotaryEmbedding(nn.Module):
    """Simplified rotary positional embedding for time."""
    
    def __init__(self, dim: int, t_max: float = 10.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (t_max ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (batch, seq_len) - timestamps in seconds
        Returns:
            (batch, seq_len, dim) - rotary embeddings
        """
        # timestamps: (batch, seq_len) -> (batch, seq_len, 1)
        t = timestamps.unsqueeze(-1)
        # (batch, seq_len, dim//2)
        freqs = t * self.inv_freq
        # (batch, seq_len, dim)
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return emb


class SimpleCrossAttention(nn.Module):
    """Simplified cross-attention layer."""
    
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
    
    def forward(
        self, 
        query: torch.Tensor,      # (B, N_q, D)
        context: torch.Tensor,    # (B, N_kv, D)
        context_mask: Optional[torch.Tensor] = None  # (B, N_kv)
    ) -> torch.Tensor:
        B, N_q, _ = query.shape
        
        query = self.norm_q(query)
        context = self.norm_kv(context)
        
        q = self.to_q(query)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, N_q, self.heads, -1).transpose(1, 2)  # (B, H, N_q, D_h)
        k = k.view(B, -1, self.heads, -1).transpose(1, 2)   # (B, H, N_kv, D_h)
        v = v.view(B, -1, self.heads, -1).transpose(1, 2)   # (B, H, N_kv, D_h)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N_q, N_kv)
        
        if context_mask is not None:
            # Expand mask for heads: (B, N_kv) -> (B, 1, 1, N_kv)
            mask = context_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        
        out = attn @ v  # (B, H, N_q, D_h)
        out = out.transpose(1, 2).reshape(B, N_q, -1)  # (B, N_q, inner_dim)
        
        return self.to_out(out)


class SimpleSelfAttention(nn.Module):
    """Simplified self-attention layer."""
    
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.heads, -1).transpose(1, 2) for t in qkv]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, -1)
        
        return self.to_out(out)


class FeedForward(nn.Module):
    """Simple feed-forward network."""
    
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimplePOYO(nn.Module):
    """
    Simplified POYO model for demonstration.
    
    Architecture:
    1. Embed spike inputs (unit embedding + time embedding)
    2. Cross-attention: latent tokens attend to inputs (compression)
    3. Self-attention: process latent representations
    4. Cross-attention: output queries attend to latents (decoding)
    5. Linear readout for final predictions
    """
    
    def __init__(
        self,
        num_units: int,
        num_sessions: int = 5,
        output_dim: int = 2,
        sequence_length: float = 1.0,
        latent_step: float = 0.1,
        num_latents_per_step: int = 8,
        dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        dim_head: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.dim = dim
        
        # Number of latent timesteps
        num_latent_steps = int(sequence_length / latent_step)
        num_latents = num_latent_steps * num_latents_per_step
        
        # Embeddings
        self.unit_emb = nn.Embedding(num_units, dim)
        self.session_emb = nn.Embedding(num_sessions, dim)
        self.latent_emb = nn.Embedding(num_latents_per_step, dim)
        
        # Time embeddings
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.time_proj = nn.Linear(dim_head, dim)
        
        # Encoder: compress inputs to latents
        self.enc_attn = SimpleCrossAttention(dim, heads, dim_head)
        self.enc_ffn = FeedForward(dim, dropout=dropout)
        
        # Processor: self-attention layers
        self.proc_layers = nn.ModuleList([
            nn.ModuleList([
                SimpleSelfAttention(dim, heads, dim_head),
                FeedForward(dim, dropout=dropout),
            ])
            for _ in range(depth)
        ])
        
        # Decoder: output queries attend to latents
        self.dec_attn = SimpleCrossAttention(dim, heads, dim_head)
        self.dec_ffn = FeedForward(dim, dropout=dropout)
        
        # Output readout
        self.readout = nn.Linear(dim, output_dim)
        
        # Precompute latent positions
        latent_times = []
        latent_indices = []
        for step in range(num_latent_steps):
            t = step * latent_step + latent_step / 2  # center of each step
            for i in range(num_latents_per_step):
                latent_times.append(t)
                latent_indices.append(i)
        
        self.register_buffer("latent_times", torch.tensor(latent_times))
        self.register_buffer("latent_indices", torch.tensor(latent_indices))
    
    def forward(
        self,
        spike_units: torch.Tensor,    # (B, N_spikes) - unit indices
        spike_times: torch.Tensor,    # (B, N_spikes) - spike timestamps  
        input_mask: torch.Tensor,     # (B, N_spikes) - True for valid spikes
        output_time: float = 0.5,     # Time at which to make prediction
    ) -> torch.Tensor:
        """
        Forward pass of SimplePOYO.
        
        Args:
            spike_units: Unit indices for each spike
            spike_times: Timestamps for each spike
            input_mask: Boolean mask for valid (non-padded) spikes
            output_time: Time at which to make the prediction
            
        Returns:
            predictions: (B, output_dim) - predicted behavior
        """
        B = spike_units.shape[0]
        device = spike_units.device
        
        # 1. Embed inputs
        # Unit embeddings
        inputs = self.unit_emb(spike_units)  # (B, N_spikes, dim)
        
        # Add time information
        time_emb = self.rotary_emb(spike_times)  # (B, N_spikes, dim_head)
        inputs = inputs + self.time_proj(time_emb)
        
        # 2. Prepare latent tokens
        latent_times = self.latent_times.unsqueeze(0).expand(B, -1)  # (B, N_latents)
        latent_indices = self.latent_indices.unsqueeze(0).expand(B, -1)  # (B, N_latents)
        
        latents = self.latent_emb(latent_indices)  # (B, N_latents, dim)
        latent_time_emb = self.rotary_emb(latent_times)
        latents = latents + self.time_proj(latent_time_emb)
        
        # 3. Encode: latents attend to inputs
        latents = latents + self.enc_attn(latents, inputs, input_mask)
        latents = latents + self.enc_ffn(latents)
        
        # 4. Process: self-attention
        for self_attn, ffn in self.proc_layers:
            latents = latents + self_attn(latents)
            latents = latents + ffn(latents)
        
        # 5. Decode: output query attends to latents
        # Create output query at the specified time
        output_time_tensor = torch.full((B, 1), output_time, device=device)
        output_query = self.session_emb(torch.zeros(B, 1, dtype=torch.long, device=device))
        output_time_emb = self.rotary_emb(output_time_tensor)
        output_query = output_query + self.time_proj(output_time_emb)
        
        output_query = output_query + self.dec_attn(output_query, latents)
        output_query = output_query + self.dec_ffn(output_query)
        
        # 6. Readout
        output = self.readout(output_query.squeeze(1))  # (B, output_dim)
        
        return output


# ============================================================================
# Training Loop
# ============================================================================

def train(
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train the SimplePOYO model on fake data."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Training on device: {device}")
    
    # Create datasets
    config = FakeRecordingConfig(
        num_units=50,
        num_samples=1000,
        sequence_length=1.0,
        output_dim=2,
    )
    
    train_dataset = FakeNeuralDataset(config, split="train", seed=42)
    val_dataset = FakeNeuralDataset(config, split="val", seed=43)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Create model
    model = SimplePOYO(
        num_units=config.num_units,
        num_sessions=5,
        output_dim=config.output_dim,
        sequence_length=config.sequence_length,
        latent_step=0.1,
        num_latents_per_step=8,
        dim=128,
        depth=2,
        heads=4,
        dim_head=32,
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            spike_times = batch["spike_times"].to(device)
            spike_units = batch["spike_units"].to(device)
            input_mask = batch["input_mask"].to(device)
            targets = batch["targets"].to(device)
            
            optimizer.zero_grad()
            
            predictions = model(spike_units, spike_times, input_mask)
            loss = F.mse_loss(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                spike_times = batch["spike_times"].to(device)
                spike_units = batch["spike_units"].to(device)
                input_mask = batch["input_mask"].to(device)
                targets = batch["targets"].to(device)
                
                predictions = model(spike_units, spike_times, input_mask)
                loss = F.mse_loss(predictions, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    logger.info(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    
    return model


if __name__ == "__main__":
    model = train()
