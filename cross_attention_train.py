import csv

import torch
import torch.nn as nn
from itertools import islice
from torch.nn import functional as F

import bpe
from bpe import get_start_and_end_tokens


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

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
model_type = 'cross-attention'#'self-attention'
max_pairs = 1_000_000
device = get_device()

torch.manual_seed(1337)


def get_self_attention_batch(my_data):
    """my_data is a single dimension very long tensor"""
    random_sample_indexes = torch.randint(len(my_data) - block_size, (batch_size,))
    x_out = torch.stack([my_data[i:i+block_size] for i in random_sample_indexes])
    y = torch.stack([my_data[i+1:i+block_size+1] for i in random_sample_indexes])
    x_out, y = x_out.to(device), y.to(device)
    return {'x_out': x_out, 'y': y}


def get_cross_attention_batch(my_data):
    """my_data is a two dimension stacked pair tensor"""
    random_sample_indexes = torch.randint(len(my_data), (batch_size,))
    x_out = torch.stack([torch.tensor(my_data[i]['x_out'][:block_size]) for i in random_sample_indexes])
    x_in = torch.stack([torch.tensor(my_data[i]['x_in'][:block_size]) for i in random_sample_indexes])
    y = torch.stack([torch.tensor(my_data[i]['x_out'][1:block_size+1]) for i in random_sample_indexes])
    x_out, x_in, y = x_out.to(device), x_in.to(device), y.to(device)
    return {'x_out': x_out, 'x_in': x_in, 'y': y}


def get_batch(my_data):
    if model_type == 'self-attention':
        return get_self_attention_batch(my_data)
    elif model_type == 'cross-attention':
        return get_cross_attention_batch(my_data)
    else:
        raise ValueError('Unknown type')


@torch.no_grad()
def estimate_loss(my_data, model):
    out = {}
    model.eval()
    for split, data_tensor in my_data.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch_dict = get_batch(data_tensor)
            logits, loss = model(**batch_dict)
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

    def forward(self, x_out, x_in=None):
        B, T, C = x_out.shape
        if x_in is not None:
            queries = self.query(x_out)
            keys = self.key(x_in)
            values = self.value(x_in)
        else:  # self attention
            queries = self.query(x_out)
            keys = self.key(x_out)
            values = self.value(x_out)

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

    def forward(self, x_out):
        B, T, C = x_out.shape
        keys = self.key(x_out)
        queries = self.query(x_out)

        wei = queries @ keys.transpose(-2, -1) * C ** -0.5  # B, T, head_size @ B, head_size, T = B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # B, T, T
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        values = self.value(x_out)
        return wei @ values  # (B, T, T) @ (B, T, head_size) = B, T, head_size


class MultiAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([HeadCross(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_out, x_in=None):
        out = torch.cat([h(x_out, x_in) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MaskedMultiAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([MaskedHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_out):
        out = torch.cat([h(x_out) for h in self.heads], dim=-1)
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

    def forward(self, x_out):
        return self.net(x_out)


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

    def forward(self, x_out, x_in=None):
        if x_in is not None:
            # Self-attention on x_in (pass only one arg, so x_in=None in HeadCross)
            x_in = x_in + self.multi_cross(self.ln1_cross(x_in))
            x_in = x_in + self.ffwd_cross(self.ln2_cross(x_in))
        else:  # self attention
            x_in = x_out

        x_out = x_out + self.masked_multi(self.ln1(x_out))
        x_out = x_out + self.multi(self.ln2(x_out), x_in)
        x_out = x_out + self.ffwd(self.ln3(x_out))
        return x_out, x_in


class CrossAttentionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Main branch embeddings
        self.emb_table = nn.Embedding(num_embeddings=bpe.out_vocab_size, embedding_dim=n_embed)
        self.position_emb_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embed)

        # Cross branch embeddings
        self.emb_table_cross = nn.Embedding(num_embeddings=bpe.in_vocab_size, embedding_dim=n_embed)
        self.position_emb_table_cross = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embed)

        self.blocks = nn.ModuleList([
            BlockCross(n_embed, n_head=n_head),
            BlockCross(n_embed, n_head=n_head),
            BlockCross(n_embed, n_head=n_head),
        ])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, bpe.out_vocab_size)

    @staticmethod
    def get_embedding(x, token_table, position_table):
        B, T = x.shape
        tok_emb = token_table(x)  # (B, T, n_embed)
        pos_emb = position_table(torch.arange(T, device=device))
        return tok_emb + pos_emb

    def forward(self, x_out, x_in=None, y=None):
        if x_in is None:
            x_in = x_out

        emb = CrossAttentionTransformer.get_embedding(x_out, token_table=self.emb_table,
                                                      position_table=self.position_emb_table)
        emb_cross = CrossAttentionTransformer.get_embedding(x_in, token_table=self.emb_table_cross,
                                                            position_table=self.position_emb_table_cross)

        # Manually iterate through blocks to pass both inputs
        block_out = emb
        block_cross_out = emb_cross
        for block in self.blocks:
            block_out, block_cross_out = block(block_out, block_cross_out)

        logits = self.lm_head(self.ln_final(block_out))  # (B, T, vocab_size)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = y.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, x_out, x_in=None, max_new_tokens=100):
        self.eval()
        tokens = get_start_and_end_tokens()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                x_window = x_out[:, -block_size:]
                logits, loss = self(x_out=x_window, x_in=x_in)
                logits = logits[:, -1, :]  # last element in T dim (B, C)
                probs = F.softmax(logits, dim=-1)
                x_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                if x_next.item() == tokens['end_out']:
                    break
                x_out = torch.cat((x_out, x_next), dim=1)  # (B, T+1)
        self.train()
        return x_out


def get_first_rows_fast(filename, max_pairs):
    with open(filename, 'r', encoding='utf-8') as f:
        # islice(iterator, stop) stops after 'max_pairs' items
        # It doesn't read a single byte more than it needs to
        first_lines = list(islice(f, max_pairs))

    # Clean up newlines
    return [line.strip() for line in first_lines]


def get_sample_val_data(num=20):
    rows_en = get_first_rows_fast('OpenSubtitles.en', max_pairs)
    rows_es = get_first_rows_fast('OpenSubtitles.es', max_pairs)

    # Split into train and validation - use 80/20 split for more training data
    n = int(train_split_ratio * len(rows_en))

    # Choose samples of validation data
    en_sentences = rows_en[n:]
    es_sentences = rows_es[n:]
    random_val_ids = torch.randint(len(en_sentences), (num,))
    en_sentences = [en_sentences[i] for i in random_val_ids]
    es_sentences = [es_sentences[i] for i in random_val_ids]

    in_merges, out_merges = bpe.get_merges()

    pairs = []
    cut_size = block_size  # 1 more token to divide it between x and y upstream on the get_batch method.
    tokens = get_start_and_end_tokens()
    for en_sent, es_sent in zip(en_sentences, es_sentences):
        en_encoded = bpe.encode(en_sent, in_merges)
        es_encoded = bpe.encode(es_sent, out_merges)
        # Truncate/pad to block_size
        en_encoded = [tokens['start_in']] + en_encoded[:cut_size] + [tokens['end_in']] * max(0, cut_size - len(en_encoded))
        es_encoded = [tokens['start_out']] + es_encoded[:cut_size] + [tokens['end_out']] * max(0, cut_size - len(es_encoded))
        pairs.append({'x_in': en_encoded, 'x_out': es_encoded})

    return pairs


def get_cross_attention_data():
    """We translate from EN to ES, ie our x_in=EN, while x_out=ES"""
    en_sentences = get_first_rows_fast('OpenSubtitles.en', max_pairs)
    es_sentences = get_first_rows_fast('OpenSubtitles.es', max_pairs)

    # Split into train and validation - use 80/20 split for more training data
    n = int(train_split_ratio * len(en_sentences))

    # Prepare translation pairs data
    in_merges, out_merges = bpe.get_merges()
    translation_pairs = []
    cut_size = block_size - 2 # 1 more token to divide it between x and y upstream on the get_batch method.
    tokens = get_start_and_end_tokens()
    for en_sent, es_sent in zip(en_sentences, es_sentences):
        en_encoded = bpe.encode(en_sent, in_merges)
        es_encoded = bpe.encode(es_sent, out_merges)
        # Truncate/pad to block_size
        en_encoded = en_encoded[:cut_size]
        en_encoded = [tokens['start_in']] + en_encoded + [tokens['end_in']] * max(1, block_size - len(en_encoded))
        es_encoded = es_encoded[:cut_size]
        es_encoded = [tokens['start_out']] + es_encoded + [tokens['end_out']] * max(1, block_size - len(es_encoded))
        #print(en_encoded)
        #print(es_encoded)
        """
        assert len(en_encoded) == 65
        assert len(es_encoded) == 65
        assert en_encoded[0] == tokens['start_in']
        assert en_encoded[-1] == tokens['end_in']
        assert es_encoded[0] == tokens['start_out']
        assert es_encoded[-1] == tokens['end_out']
        """
        translation_pairs.append({'x_in': en_encoded, 'x_out': es_encoded})

    train_pairs = translation_pairs[:n]
    val_pairs = translation_pairs[n:]
    print(f'Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}')
    return {
        'train': train_pairs,
        'val': val_pairs
    }, out_merges, in_merges


def get_self_attention_data():
    with open('shakespear.txt', 'r') as f:
        text = f.read()

    tokens = list([int(i) for i in text.encode('utf-8')])
    print(f'length tokens: {len(tokens)}')
    print(f'length text: {len(text)}')

    merges = bpe.train_merges(tokens, bpe.out_vocab_size)

    # Justa a long 1 d tensor. bah!
    encoded_data = torch.tensor(bpe.encode(text, merges), dtype=torch.long)
    n = int(train_split_ratio * len(encoded_data))
    return {
        'train': encoded_data[:n],
        'val': encoded_data[n:]
    }, merges, {}


def get_data_and_merges():
    if model_type == 'self-attention':
        return get_self_attention_data()
    elif model_type == 'cross-attention':
        return get_cross_attention_data()
    else:
        raise Exception(f'Unknown type {model_type}')


def train():
    data, out_merges, in_merges = get_data_and_merges()

    model = CrossAttentionTransformer()
    model = model.to(device)
    try:
        model.load_state_dict(torch.load('cross_attention_model.pth'))
        print('training old model')
        model.train()
    except FileNotFoundError:
        print('Creating model from scratch')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(data, model)
            print(f"step {iter} train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
        batch_dict = get_batch(data['train'])
        logits, loss = model(**batch_dict)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the model after training
    torch.save(model.state_dict(), 'cross_attention_model.pth')
    print(f"Model saved to cross_attention_model.pth")


if __name__ == '__main__':
    train()
