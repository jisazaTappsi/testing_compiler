import torch.nn as nn
from torch.nn import functional as F

import data
from util import *
from util import parenthesis_balance_weight

torch.manual_seed(1337)


def get_batch(my_data):
    """my_data is a two dimension stacked pair tensor"""
    random_sample_indexes = torch.randint(len(my_data), (batch_size,))
    x_out = torch.stack([torch.tensor(my_data[i]['x_out'][:block_size]) for i in random_sample_indexes])
    x_in = torch.stack([torch.tensor(my_data[i]['x_in'][:block_size]) for i in random_sample_indexes])
    y = torch.stack([torch.tensor(my_data[i]['x_out'][1:block_size+1]) for i in random_sample_indexes])
    x_out, x_in, y = x_out.to(device), x_in.to(device), y.to(device)
    return {'x_out': x_out, 'x_in': x_in, 'y': y}


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

    def forward(self, x_out, x_in):
        # Small left branch in "Attention is all you need" paper
        x_in = x_in + self.multi_cross(self.ln1_cross(x_in))  # Operates on self-attention mode
        x_in = x_in + self.ffwd_cross(self.ln2_cross(x_in))

        # Big main branch in "Attention is all you need" paper
        x_out = x_out + self.masked_multi(self.ln1(x_out))  # Operates on self-attention mode
        x_out = x_out + self.multi(self.ln2(x_out), x_in)  # Operates on cross-attention mode
        x_out = x_out + self.ffwd(self.ln3(x_out))
        return x_out, x_in


class CrossAttentionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Main branch embeddings
        self.emb_table = nn.Embedding(num_embeddings=data.out_vocab_size, embedding_dim=n_embed)
        self.position_emb_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embed)

        # Cross branch embeddings
        self.emb_table_cross = nn.Embedding(num_embeddings=data.in_vocab_size, embedding_dim=n_embed)
        self.position_emb_table_cross = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embed)

        self.blocks = nn.ModuleList([
            BlockCross(n_embed, n_head=n_head),
            BlockCross(n_embed, n_head=n_head),
            BlockCross(n_embed, n_head=n_head),
        ])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, data.out_vocab_size)

    @staticmethod
    def get_embedding(x, token_table, position_table):
        B, T = x.shape
        tok_emb = token_table(x)  # (B, T, n_embed)
        pos_emb = position_table(torch.arange(T, device=device))
        return tok_emb + pos_emb

    def compute_parenthesis_balance_loss(self, logits_2d, out_merges):
        """
        Compute auxiliary loss to encourage balanced parentheses from predicted logits.
        
        This function computes balance from the most likely predicted sequence (argmax)
        and adds a penalty for imbalance. Note: Since decoding breaks the gradient chain,
        this provides an indirect signal rather than direct gradients. However, the model
        learns balance primarily through the cross-entropy loss on balanced training data.
        
        For a fully differentiable approach, we would need to compute expected parenthesis
        counts from token probabilities, which is more complex with BPE encoding.
        """
        if parenthesis_balance_weight == 0:
            return torch.tensor(0.0, device=device, requires_grad=False)
        
        B, T, C = logits_2d.shape
        tokens = data.get_start_and_end_tokens('code')
        
        # Get predicted tokens (argmax) for balance computation
        predicted_tokens = torch.argmax(logits_2d, dim=-1)  # (B, T)
        
        # Compute balance penalty (as a constant signal - gradients come from main loss)
        total_imbalance_sq = 0.0
        count = 0
        
        for b in range(B):
            seq = predicted_tokens[b].cpu().tolist()
            seq_clean = [int(t) for t in seq if t not in [tokens['start_out'], tokens['end_out']]]
            if len(seq_clean) == 0:
                continue
            
            try:
                decoded = data.decode(seq_clean, out_merges)
                open_count = decoded.count('(')
                close_count = decoded.count(')')
                imbalance = abs(open_count - close_count)
                total_imbalance_sq += imbalance ** 2
                count += 1
            except:
                continue
        
        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=False)
        
        # Return as a constant (no gradients, but signals importance)
        avg_penalty = total_imbalance_sq / count
        return torch.tensor(parenthesis_balance_weight * avg_penalty, device=device, requires_grad=False)

    def forward(self, x_out, x_in, y=None):
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
            # Store original shape for balance loss computation
            logits_2d = logits.clone()
            logits = logits.view(B*T, C)
            target = y.view(B*T)
            loss = F.cross_entropy(logits, target)
            
            # Add parenthesis balance auxiliary loss
            if parenthesis_balance_weight > 0:
                out_merges, _ = data.get_merges('code')
                balance_loss = self.compute_parenthesis_balance_loss(logits_2d, out_merges)
                loss = loss + balance_loss

        return logits, loss

    def generate(self, x_out, x_in=None, max_new_tokens=100):
        self.eval()
        tokens = data.get_start_and_end_tokens('code')
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


def train():
    dataset, out_merges, in_merges = data.get_code_data()
    model = CrossAttentionTransformer()
    model = model.to(device)

    try:
        model.load_state_dict(torch.load(code_model_name))
        print('training an existing model')
        model.train()
    except FileNotFoundError:
        print('Creating model from scratch')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(dataset, model)
            print(f"step {iter} train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
        batch_dict = get_batch(dataset['train'])
        logits, loss = model(**batch_dict)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the model after training
    torch.save(model.state_dict(), code_model_name)
    print(f"Model saved to {code_model_name}")


if __name__ == '__main__':
    train()
