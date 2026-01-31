from util import *
from lang_train import CrossAttentionTransformer

data_to_params_ratio = 20
char_token_compression = 7
avg_sentence_length = block_size / 2
avg_token_per_sentence = avg_sentence_length/char_token_compression


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Create model instance
model = CrossAttentionTransformer()
model = model.to(device)

# Load the saved state dict
model.load_state_dict(torch.load(code_model_name))
model.eval()

param_count = count_parameters(model)
token_count = data_to_params_ratio * param_count

print(f'total params: {param_count}')
print(f'Should train on {token_count} tokens')
print(f'Should train on {round(token_count/avg_token_per_sentence)} sentences')

rows = get_first_rows_fast(dataset_name, 100_000)

import data
import statistics

in_merges, out_merges = data.get_merges('code')
lens0 = [len(data.encode(r.split(',')[0], in_merges)) for r in rows]
lens1 = [len(data.encode(r.split(',')[1], in_merges)) for r in rows]

print(f'stats are: {statistics.mean(lens0)=}, {statistics.stdev(lens0)=}, {max(lens0)=}, {min(lens0)=}')
print(f'stats are: {statistics.mean(lens1)=}, {statistics.stdev(lens1)=}, {max(lens1)=}, {min(lens1)=}')

# cut below which 99.7% of samples lie.
above_3_stdev = statistics.mean(lens1) + 3*statistics.stdev(lens1)
print(f'block_size should be: {min(above_3_stdev, max(lens1))}')
