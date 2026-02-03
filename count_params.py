import data
import statistics

from util import *
from code_train import CrossAttentionTransformer

data_to_params_ratio = 20


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
rows = get_first_rows_fast(dataset_name, 10_000)

lex_merges, _ = data.get_merges()
lens0 = [len(data.encode(r.split(',')[0], lex_merges)) for r in rows]
lens1 = [len(data.encode(r.split(',')[1], lex_merges)) for r in rows]
avg_tokens_per_sentence = statistics.mean(lens1)

print(f'total params: {param_count}')
print(f'Should train on {token_count} tokens')
print(f'Should train on {round(token_count/avg_tokens_per_sentence)} sentences')
print(f'stats are: {statistics.mean(lens0)=}, {statistics.stdev(lens0)=}, {max(lens0)=}, {min(lens0)=}')
print(f'stats are: {avg_tokens_per_sentence=}, {statistics.stdev(lens1)=}, {max(lens1)=}, {min(lens1)=}')

# cut below which 99.7% of samples lie.
above_3_std_dev = avg_tokens_per_sentence + 3*statistics.stdev(lens1)
print(f'block_size should be: {min(above_3_std_dev, max(lens1))}')
