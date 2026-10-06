# POYO Simple Example

A minimal, self-contained example demonstrating the POYO architecture with synthetic data.

## Overview

This example creates a simplified version of POYO that:
1. **Generates fake neural spike data** - No real datasets required
2. **Implements core POYO components** - Cross-attention, self-attention, rotary embeddings
3. **Trains end-to-end** - Complete training loop with validation

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SimplePOYO                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   INPUT SPIKES                                                           │
│   ┌──────────────────┐                                                   │
│   │ spike_units      │ ──► Unit Embedding ──┐                           │
│   │ spike_times      │ ──► Time Embedding ──┼──► inputs (B, N, D)       │
│   └──────────────────┘                      │                            │
│                                             │                            │
│   LATENT TOKENS                             │                            │
│   ┌──────────────────┐                      │                            │
│   │ latent_indices   │ ──► Latent Emb ─────┐│                            │
│   │ latent_times     │ ──► Time Emb ───────┼┼──► latents (B, L, D)      │
│   └──────────────────┘                     ││                            │
│                                            ││                            │
│   ┌────────────────────────────────────────┼┼───────────────────────────┤
│   │              ENCODER (Cross-Attention) ││                            │
│   │  latents ◄─── attend to ───► inputs    ││                            │
│   │  (compress spike sequence into latents)││                            │
│   └────────────────────────────────────────┼┼───────────────────────────┤
│                                            ││                            │
│   ┌────────────────────────────────────────┼┼───────────────────────────┤
│   │              PROCESSOR (Self-Attention)││                            │
│   │  latents ◄─── attend to ───► latents   ││                            │
│   │  (refine latent representations)       ││                            │
│   └────────────────────────────────────────┼┼───────────────────────────┤
│                                            ││                            │
│   OUTPUT QUERY                             ││                            │
│   ┌──────────────────┐                     ││                            │
│   │ output_time      │ ──► Time Emb ───────┼┼──► query (B, 1, D)        │
│   │ session_idx      │ ──► Session Emb ────┘│                            │
│   └──────────────────┘                      │                            │
│                                             │                            │
│   ┌────────────────────────────────────────────────────────────────────┤
│   │              DECODER (Cross-Attention)                              │
│   │  query ◄─── attend to ───► latents                                  │
│   │  (extract prediction at output_time)                                │
│   └────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────┐                                                   │
│   │ Linear Readout   │ ──► predictions (B, output_dim)                  │
│   └──────────────────┘                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Fake Dataset (`FakeNeuralDataset`)
- Generates synthetic Poisson spike trains
- Creates random behavior targets (e.g., 2D velocity)
- Mimics the structure of real neural recording data

### 2. SimplePOYO Model
- **Unit Embedding**: Maps neuron IDs to learned vectors
- **Rotary Time Embedding**: Encodes spike timestamps
- **Latent Tokens**: Fixed grid of learnable tokens that compress input
- **Cross-Attention (Encoder)**: Latents attend to inputs
- **Self-Attention (Processor)**: Latents attend to themselves
- **Cross-Attention (Decoder)**: Output query attends to latents
- **Linear Readout**: Final prediction layer

## Usage

### Basic Training

```bash
cd /pscratch/sd/a/ahhyun/EcoGFound/POYO/torch_brain/examples/poyo_simple
python train_simple.py
```

### Custom Configuration

```python
from train_simple import train, FakeRecordingConfig

# Custom dataset config
config = FakeRecordingConfig(
    num_units=100,        # More neurons
    num_samples=5000,     # More training samples
    sequence_length=2.0,  # Longer windows
    output_dim=3,         # 3D output
)

# Train with custom hyperparameters
model = train(
    num_epochs=100,
    batch_size=64,
    lr=5e-4,
)
```

## Comparison with Full POYO+

| Feature | SimplePOYO | POYO+ |
|---------|------------|-------|
| Time Embedding | Rotary (simplified) | Rotary (full) |
| Cross-Attention | Standard | Rotary with value rotation |
| Multi-task | Single output | Multiple task heads |
| Collation | Simple padding | `pad8()` + `chain()` |
| Dataset | Fake/synthetic | Real neural recordings |
| Vocabulary | Fixed size | `InfiniteVocabEmbedding` |

## Data Flow

```
                        TRAINING
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   FakeNeuralDataset                                          │
│   ├── spike_times: [0.1, 0.3, 0.5, ...] (sorted)            │
│   ├── spike_units: [0, 5, 2, ...] (neuron IDs)              │
│   └── target: [vx, vy] (behavior)                           │
│                     │                                        │
│                     ▼                                        │
│   collate_fn                                                 │
│   ├── Pad to max length (multiple of 8)                     │
│   ├── Create input_mask (True for valid spikes)             │
│   └── Stack targets                                          │
│                     │                                        │
│                     ▼                                        │
│   SimplePOYO.forward()                                       │
│   ├── Embed inputs + time                                    │
│   ├── Prepare latent grid                                    │
│   ├── Encoder cross-attention                                │
│   ├── Processor self-attention (x depth)                     │
│   ├── Decoder cross-attention                                │
│   └── Linear readout                                         │
│                     │                                        │
│                     ▼                                        │
│   MSE Loss: ||predictions - targets||²                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Files

```
poyo_simple/
├── README.md           # This file
├── train_simple.py     # Main training script with fake dataset
└── configs/
    └── (optional YAML configs if using Hydra)
```

## Expected Output

```
INFO:__main__:Training on device: cuda
INFO:__main__:Model parameters: 295,298
INFO:__main__:Epoch   1/50 | Train Loss: 0.2834 | Val Loss: 0.2756 | LR: 0.001000
INFO:__main__:Epoch  10/50 | Train Loss: 0.2512 | Val Loss: 0.2543 | LR: 0.000905
INFO:__main__:Epoch  20/50 | Train Loss: 0.2398 | Val Loss: 0.2467 | LR: 0.000655
INFO:__main__:Epoch  30/50 | Train Loss: 0.2301 | Val Loss: 0.2389 | LR: 0.000345
INFO:__main__:Epoch  40/50 | Train Loss: 0.2245 | Val Loss: 0.2356 | LR: 0.000095
INFO:__main__:Epoch  50/50 | Train Loss: 0.2234 | Val Loss: 0.2348 | LR: 0.000000
INFO:__main__:Training complete! Best validation loss: 0.2348
```

## Next Steps

After understanding this simple example, you can:

1. **Use real data**: Replace `FakeNeuralDataset` with `torch_brain.data.Dataset`
2. **Add multi-task**: Use `MultitaskReadout` for multiple prediction heads
3. **Use full POYO+**: See `examples/poyo_plus/` for the complete implementation
