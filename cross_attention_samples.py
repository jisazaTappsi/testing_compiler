import bpa
import torch
from cross_attention_train import CrossAttentionTransformer, get_sample_val_data, block_size
from bpa import get_start_and_end_tokens

# Set up device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


def run():
    # Load data and merges
    val_samples = get_sample_val_data(num=100)
    in_merges, out_merges = bpa.get_merges()

    # Create model instance
    model = CrossAttentionTransformer()
    model = model.to(device)

    # Load the saved state dict
    model.load_state_dict(torch.load('cross_attention_model.pth'))
    model.eval()

    # Run generation
    tokens = get_start_and_end_tokens()
    for row in val_samples:
        context = torch.tensor([[tokens['start_out']]], dtype=torch.long, device=device)
        data_in = row['x_in']
        print(f'I: {bpa.decode([e for e in data_in if e != tokens['end_in']], in_merges)}')
        translation = bpa.decode(
            model.generate(x_out=context,
                           x_in=torch.tensor([data_in], dtype=torch.long, device=device),
                           max_new_tokens=block_size)[0].tolist(),
            out_merges
        )
        print(f'O: {translation}\n')


if __name__ == '__main__':
    run()
