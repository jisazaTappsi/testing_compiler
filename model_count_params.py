import torch

from util import *
from cross_attention_train import CrossAttentionTransformer, model_name

data_to_params_ratio = 20
avg_token_per_sentence = 6.5


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Create model instance
model = CrossAttentionTransformer()
model = model.to(device)

# Load the saved state dict
model.load_state_dict(torch.load(model_name))
model.eval()

param_count = count_parameters(model)
token_count = data_to_params_ratio * param_count

print(f'total params: {param_count}')
print(f'Should train on {token_count} tokens')
print(f'Should train on {round(token_count/avg_token_per_sentence)} sentences')
