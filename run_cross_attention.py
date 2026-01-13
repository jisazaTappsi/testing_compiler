import bpa
import torch
from train_cross_attention import CrossAttentionTransformer, get_cross_attention_data

# Set up device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Load data and merges
data, out_merges, in_merges = get_cross_attention_data()
data_val = data['val']

# Create model instance
model = CrossAttentionTransformer()
model = model.to(device)

# Load the saved state dict
model.load_state_dict(torch.load('cross_attention_model.pth'))
model.eval()

# Run generation
for idx in torch.randint(len(data['val']), (20, )):
    context = torch.tensor([[ord(' ')]], dtype=torch.long, device=device)
    data_in = data_val[idx]['x_in']
    print(f'Original in English: {bpa.decode(data_in, in_merges)}')
    translation = bpa.decode(
        model.generate(x_out=context,
                       x_in=torch.tensor([data_in], dtype=torch.long, device=device),
                       max_new_tokens=400)[0].tolist(),
        out_merges
    )
    print(f'Translation in Franchute: {translation}' )
