"""
Super fast test config - 5 minute training
"""
out_dir = 'out-quantara-test'
eval_interval = 50
eval_iters = 10  # Very few eval iterations
log_interval = 10
always_save_checkpoint = True

wandb_log = False
dataset = 'quantara_emotion'

gradient_accumulation_steps = 1
batch_size = 8
block_size = 128

# Tiny model
n_layer = 2
n_head = 2
n_embd = 128
dropout = 0.1

learning_rate = 1e-3
max_iters = 500
lr_decay_iters = 500
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 50

device = 'cpu'
compile = False
