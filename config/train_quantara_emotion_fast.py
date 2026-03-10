"""
===============================================================================
QUANTARA NANOGPT - Fast Emotion Training (Debug/Test)
===============================================================================
Quick training config for testing on MacBook.
Trains a small model in ~15-30 minutes.

Usage:
  python train.py config/train_quantara_emotion_fast.py --device=mps
  python train.py config/train_quantara_emotion_fast.py --device=cpu --compile=False
===============================================================================
"""

out_dir = 'out-quantara-emotion-fast'

eval_interval = 250
eval_iters = 20  # Reduced for faster CPU training
log_interval = 10  # More frequent logging
always_save_checkpoint = False

wandb_log = False
wandb_project = 'quantara-emotion'
wandb_run_name = 'emotion-gpt-fast'

dataset = 'quantara_emotion'

# Smaller batch for quick iteration
gradient_accumulation_steps = 1
batch_size = 16
block_size = 256  # Shorter context

# Tiny model for fast training
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.2

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# MacBook defaults (CPU - MPS not available in this PyTorch build)
device = 'cpu'
compile = False
