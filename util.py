from itertools import islice

import torch

batch_size = 64 # 32
block_size = 64 # 256
max_iters = 5_000
eval_interval = 500
learning_rate = 3e-4 # 1e-4
eval_iters = 200
dropout = 0.2
n_head = 6  # 4
n_embed = 64 * n_head  # 32
train_split_ratio = 0.8
max_pairs = 10_000
lang_model_name = 'lang_model.pth'

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


def get_first_rows_fast(filename, max_pairs):
    with open(filename, 'r', encoding='utf-8') as f:
        first_lines = list(islice(f, max_pairs))
    return [line.strip() for line in first_lines]
