"""
===============================================================================
QUANTARA NANOGPT - Medium Emotion Training (32 Emotions)
===============================================================================
Balanced training config — stronger model than fast, feasible on CPU.
Trains in ~2-4 hours on MacBook CPU.

Usage:
  python train.py config/train_quantara_emotion_medium.py
  python train.py config/train_quantara_emotion_medium.py --device=mps
===============================================================================
"""

out_dir = 'out-quantara-emotion-medium'

eval_interval = 500
eval_iters = 50
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'quantara-emotion'
wandb_run_name = 'emotion-gpt-medium'

dataset = 'quantara_emotion'

gradient_accumulation_steps = 2
batch_size = 24
block_size = 384  # Longer context for biometric-enriched text

# Medium model — 2x the fast config
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.15

learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 6e-5
beta2 = 0.95
weight_decay = 0.1

warmup_iters = 300

# MacBook defaults
device = 'cpu'
compile = False
