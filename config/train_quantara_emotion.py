"""
===============================================================================
QUANTARA NANOGPT - Emotion Training Configuration (32 Emotions)
===============================================================================
Train an emotion-aware GPT model for the Quantara Neural Ecosystem.
Supports 32 emotions across 9 families with hierarchical classification.

Integrates with:
- AI Conversational Coach
- Emotion-Aware Training Engine
- Psychology Emotion Database
- Neural Workflow AI Engine

Usage:
  # MacBook (MPS - Apple Silicon)
  python train.py config/train_quantara_emotion.py

  # MacBook (CPU only)
  python train.py config/train_quantara_emotion.py --device=cpu --compile=False

  # GPU Server
  python train.py config/train_quantara_emotion.py --device=cuda
===============================================================================
"""

# Output directory
out_dir = 'out-quantara-emotion'

# Evaluation settings
eval_interval = 500      # Evaluate every N steps
eval_iters = 200         # Number of iterations for evaluation
log_interval = 50        # Print logs every N steps

# Checkpointing
always_save_checkpoint = True  # Save model at each eval

# Weights & Biases logging (optional)
wandb_log = False
wandb_project = 'quantara-emotion'
wandb_run_name = 'emotion-gpt-v1'

# Dataset
dataset = 'quantara_emotion'

# Training hyperparameters
gradient_accumulation_steps = 4   # Simulate larger batch on small GPU
batch_size = 32                   # Samples per GPU
block_size = 512                  # Context window (emotion + text)

# Model architecture (Medium size for emotion understanding)
n_layer = 8          # 8 transformer layers
n_head = 8           # 8 attention heads
n_embd = 512         # 512 embedding dimension
dropout = 0.15       # Slightly lower dropout for larger dataset

# Optimizer settings
learning_rate = 5e-4          # Good for medium models
max_iters = 10000             # Train for 10K iterations
lr_decay_iters = 10000        # Linear decay over training
min_lr = 5e-5                 # Final learning rate
beta2 = 0.95                  # Adam beta2
weight_decay = 0.1            # L2 regularization

# Learning rate warmup
warmup_iters = 500

# For Apple Silicon MacBooks (uncomment if needed)
# device = 'mps'      # Use Metal Performance Shaders
# compile = False     # MPS doesn't support torch.compile yet

# For CPU-only training (uncomment if needed)
# device = 'cpu'
# compile = False
