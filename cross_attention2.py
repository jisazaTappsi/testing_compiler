import torch
import torch.nn as nn
from torch.nn import functional as F

import bpa

batch_size = 32 # 64
block_size = 8 # 256
max_iters = 1_000
eval_interval = 500
learning_rate = 1e-3 # 3e-4
eval_iters = 200
dropout = 0.2
n_head = 4  # 6
n_embed = 64  # 64 * n_head
vocab_size =  256 # 10_000
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


torch.manual_seed(1337)

with open('shakespear.txt', 'r') as f:
    text = f.read()

tokens = list([int(i) for i in text.encode('utf-8')])
print(f'length tokens: {len(tokens)}')
print(f'length text: {len(text)}')

merges = bpa.train_merges(tokens, vocab_size)


data = torch.tensor(bpa.encode(text, merges), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]  # justa a long 1 d tensor. bah!
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    random_sample_indexes = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in random_sample_indexes])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_sample_indexes])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(x=X, y=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class HeadCross(nn.Module):
    """one head self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_cross=None):
        B, T, C = x.shape
        if x_cross is not None:
            queries = self.query(x)
            keys = self.key(x_cross)
            values = self.value(x_cross)
        else:  # self attention
            queries = self.query(x)
            keys = self.key(x)
            values = self.value(x)

        wei = queries @ keys.transpose(-2, -1) * C ** -0.5  # B, T, head_size @ B, head_size, T = B, T, T
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ values  # (B, T, T) @ (B, T, head_size) = B, T, head_size


class MaskedHead(nn.Module):
    """one head self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril',
                             torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        keys = self.key(x)
        queries = self.query(x)

        wei = queries @ keys.transpose(-2, -1) * C ** -0.5  # B, T, head_size @ B, head_size, T = B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # B, T, T
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        values = self.value(x)
        return wei @ values  # (B, T, T) @ (B, T, head_size) = B, T, head_size


class MultiAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([HeadCross(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_cross=None):
        out = torch.cat([h(x, x_cross) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MaskedMultiAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([MaskedHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BlockCross(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.masked_multi = MaskedMultiAttention(n_head, head_size=n_embed//n_head)
        self.ln2 = nn.LayerNorm(n_embed)
        self.multi = MultiAttention(n_head, head_size=n_embed//n_head)
        self.ln3 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)

        self.ln1_cross = nn.LayerNorm(n_embed)
        self.multi_cross = MultiAttention(n_head, head_size=n_embed//n_head)
        self.ln2_cross = nn.LayerNorm(n_embed)
        self.ffwd_cross = FeedForward(n_embed)

    def forward(self, x, x_cross=None):
        if x_cross is not None:
            # Self-attention on x_cross (pass only one arg, so x_cross=None in HeadCross)
            x_cross = x_cross + self.multi_cross(self.ln1_cross(x_cross))
            x_cross = x_cross + self.ffwd_cross(self.ln2_cross(x_cross))
        else:  # self attention
            x_cross = x

        x = x + self.masked_multi(self.ln1(x))
        x = x + self.multi(self.ln2(x), x_cross)
        x = x + self.ffwd(self.ln3(x))
        return x, x_cross

class CrossAttentionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Main branch embeddings
        self.emb_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed)
        self.position_emb_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embed)

        # Cross branch embeddings
        self.emb_table_cross = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed)
        self.position_emb_table_cross = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embed)

        self.blocks = nn.ModuleList([
            BlockCross(n_embed, n_head=n_head),
            BlockCross(n_embed, n_head=n_head),
            BlockCross(n_embed, n_head=n_head),
        ])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    @staticmethod
    def get_embedding(x, token_table, position_table):
        B, T = x.shape
        tok_emb = token_table(x)  # (B, T, n_embed)
        pos_emb = position_table(torch.arange(T, device=device))
        return tok_emb + pos_emb

    def forward(self, x, x_cross=None, y=None):
        if x_cross is None:
            x_cross = x

        emb = CrossAttentionTransformer.get_embedding(x, token_table=self.emb_table,
                                                      position_table=self.position_emb_table)
        emb_cross = CrossAttentionTransformer.get_embedding(x_cross, token_table=self.emb_table_cross,
                                                            position_table=self.position_emb_table_cross)

        # Manually iterate through blocks to pass both inputs
        x_out = emb
        x_cross_out = emb_cross
        for block in self.blocks:
            x_out, x_cross_out = block(x_out, x_cross_out)

        logits = self.lm_head(self.ln_final(x_out))  # (B, T, vocab_size)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = y.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, x, x_cross=None, max_new_tokens=100):
        # If no cross input provided, use the same as main input
        if x_cross is None:
            x_cross = x

        for _ in range(max_new_tokens):
            x_window = x[:, -block_size:]
            x_cross_window = x_cross[:, -block_size:]
            logits, loss = self(x=x_window, x_cross=x_cross_window)
            logits = logits[:, -1, :]  # last element in T dim (B, C)
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            x = torch.cat((x, x_next), dim=1)  # (B, T+1)
            x_cross = torch.cat((x_cross, x_next), dim=1)  # (B, T+1)
        return x


model = CrossAttentionTransformer()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter} train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(x=xb, y=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.tensor([[ord(' ')]], dtype=torch.long, device=device)
print(bpa.decode(model.generate(context, max_new_tokens=2_000)[0].tolist(), merges))
