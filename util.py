import torch

batch_size = 64  # 32
block_size = 64  # 256
max_iters = 30_001
eval_interval = 500
lr_peak = 8e-4  # peak learning rate
lr_min = 1e-4  # min learning rate
warmup_iters = 0.03 * max_iters   # linear warmup steps (e.g. ~3% of max_iters)
eval_iters = 200
dropout = 0.2
n_head = 3  # 4
n_embed = 64 * n_head  # 32
train_split_ratio = 0.8
max_samples = 600_000
introduce_error = False

if introduce_error:
    code_model_name = 'model_error.pth'
    dataset_name = 'dataset_error.pkl'
else:
    code_model_name = 'code_model.pth'
    dataset_name = 'dataset.pkl'


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
