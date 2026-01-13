import bpa
import torch
from train_cross_attention import CrossAttentionTransformer, get_sample_val_data, get_eof

# Set up device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


def run():
    # Load data and merges
    val_samples = get_sample_val_data(num=10)
    in_merges, out_merges = bpa.get_merges()

    # Create model instance
    model = CrossAttentionTransformer()
    model = model.to(device)

    # Load the saved state dict
    model.load_state_dict(torch.load('cross_attention_model.pth'))
    model.eval()

    # Run generation
    eof = get_eof()
    for row in val_samples:
        context = torch.tensor([[ord(' ')]], dtype=torch.long, device=device)
        data_in = row['x_in']
        print(f'Original in English: {bpa.decode([e for e in data_in if e != eof['in']], in_merges)}')
        translation = bpa.decode(
            model.generate(x_out=context,
                           x_in=torch.tensor([data_in], dtype=torch.long, device=device),
                           max_new_tokens=400)[0].tolist(),
            out_merges
        )
        print(f'Translation in Franchute: {translation}' )


if __name__ == '__main__':
    run()
