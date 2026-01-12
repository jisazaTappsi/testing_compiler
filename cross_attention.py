import csv
import bpa
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # 64
block_size = 8 # 256
max_iters = 5_000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
dropout = 0.3
weight_decay = 1e-4  # L2 regularization
label_smoothing = 0.1  # Regularization
n_head = 4  # 6
n_embed = 32  # 64 * n_head
en_vocab_size = 500  # Source (English) vocabulary size
fr_vocab_size = 500  # Target (French) vocabulary size
max_pairs = 5_000

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


torch.manual_seed(1337)

# Load translation pairs separately - use more data to reduce overfitting
with open('en_fr_translation.csv', mode='r', newline='') as csv_file:
    translation_corpus = csv.reader(csv_file, delimiter=',')
    rows = list(translation_corpus)[1:bpa.max_pairs]
    en_sentences = [row[0] for row in rows if len(row) >= 2]
    fr_sentences = [row[1] for row in rows if len(row) >= 2]
    
    # Create separate text corpora for English and French
    en_text = ' '.join(en_sentences)
    fr_text = ' '.join(fr_sentences)


# Train separate tokenizers for source (English) and target (French)
en_tokens = list[int]([int(i) for i in en_text.encode('utf-8')])
fr_tokens = list[int]([int(i) for i in fr_text.encode('utf-8')])
print(f'English tokens length: {len(en_tokens)}')
print(f'French tokens length: {len(fr_tokens)}')

en_merges = bpa.train_merges(en_tokens, target_vocab_size=en_vocab_size)
fr_merges = bpa.train_merges(fr_tokens, target_vocab_size=fr_vocab_size)


# Prepare translation pairs data
translation_pairs = []
for en_sent, fr_sent in zip(en_sentences, fr_sentences):
    en_encoded = bpa.encode(en_sent, en_merges)
    fr_encoded = bpa.encode(fr_sent, fr_merges)
    # Truncate/pad to block_size
    en_encoded = en_encoded[:block_size] + [0] * max(0, block_size - len(en_encoded))
    fr_encoded = fr_encoded[:block_size] + [0] * max(0, block_size - len(fr_encoded))
    translation_pairs.append((en_encoded[:block_size], fr_encoded[:block_size]))

# Split into train and validation - use 80/20 split for more training data
n = int(0.8 * len(translation_pairs))
train_pairs = translation_pairs[:n]
val_pairs = translation_pairs[n:]
print(f'Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}')

def get_batch(split):
    pairs = train_pairs if split == 'train' else val_pairs
    # Sample random pairs
    indices = torch.randint(len(pairs), (batch_size,))
    src_batch = []
    tgt_batch = []
    tgt_shifted_batch = []
    
    for idx in indices:
        src, tgt = pairs[idx]
        src_batch.append(src)
        tgt_batch.append(tgt)
        # For teacher forcing: target input is shifted (prepend SOS or use previous tokens)
        tgt_shifted = [0] + tgt[:-1]  # Simple shift, 0 acts as SOS token
        tgt_shifted_batch.append(tgt_shifted)
    
    src_tensor = torch.tensor(src_batch, dtype=torch.long, device=device)
    tgt_tensor = torch.tensor(tgt_shifted_batch, dtype=torch.long, device=device)
    tgt_labels = torch.tensor(tgt_batch, dtype=torch.long, device=device)
    
    return src_tensor, tgt_tensor, tgt_labels


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            src, tgt, tgt_labels = get_batch(split)
            logits, loss = model(src, tgt, tgt_labels)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class SelfAttentionHead(nn.Module):
    """one head self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril',
                             torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        keys = self.key(x)
        queries = self.query(x)

        wei = queries @ keys.transpose(-2, -1) * C ** -0.5  # B, T, head_size @ B, head_size, T = B, T, T
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float("-inf"))
        else:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # B, T, T
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        values = self.value(x)
        return wei @ values  # (B, T, T) @ (B, T, head_size) = B, T, head_size


class CrossAttentionHead(nn.Module):
    """one head cross attention - decoder queries attend to encoder keys/values"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_x, encoder_x):
        """
        decoder_x: (B, T_dec, C) - decoder input (queries come from here)
        encoder_x: (B, T_enc, C) - encoder output (keys and values come from here)
        """
        B, T_dec, C = decoder_x.shape
        T_enc = encoder_x.shape[1]
        
        queries = self.query(decoder_x)  # (B, T_dec, head_size)
        keys = self.key(encoder_x)       # (B, T_enc, head_size)
        values = self.value(encoder_x)   # (B, T_enc, head_size)
        
        # Cross-attention: decoder queries attend to encoder keys
        wei = queries @ keys.transpose(-2, -1) * C ** -0.5  # (B, T_dec, T_enc)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ values  # (B, T_dec, T_enc) @ (B, T_enc, head_size) = (B, T_dec, head_size)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_x, encoder_x):
        """
        decoder_x: (B, T_dec, C) - decoder input
        encoder_x: (B, T_enc, C) - encoder output
        """
        out = torch.cat([h(decoder_x, encoder_x) for h in self.heads], dim=-1)
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


class EncoderBlock(nn.Module):
    """Encoder block: self-attention + feedforward"""
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.sa = MultiHeadSelfAttention(n_head, head_size=n_embed//n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """Decoder block: masked self-attention + cross-attention + feedforward"""
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.sa = MultiHeadSelfAttention(n_head, head_size=n_embed//n_head)
        self.ca = MultiHeadCrossAttention(n_head, head_size=n_embed//n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ln3 = nn.LayerNorm(n_embed)

    def forward(self, decoder_x, encoder_x):
        # Masked self-attention (decoder can only see previous tokens)
        decoder_x = decoder_x + self.sa(self.ln1(decoder_x))
        # Cross-attention (decoder attends to encoder)
        decoder_x = decoder_x + self.ca(self.ln2(decoder_x), encoder_x)
        # Feedforward
        decoder_x = decoder_x + self.ffwd(self.ln3(decoder_x))
        return decoder_x

class EncoderDecoderModel(nn.Module):
    """Encoder-Decoder model for translation"""
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        # Source (English) embeddings
        self.src_token_embedding = nn.Embedding(en_vocab_size, n_embed)
        self.src_position_embedding = nn.Embedding(block_size, n_embed)
        
        # Target (French) embeddings
        self.tgt_token_embedding = nn.Embedding(fr_vocab_size, n_embed)
        self.tgt_position_embedding = nn.Embedding(block_size, n_embed)
        
        # Encoder: processes source language
        self.encoder_blocks = nn.Sequential(
            EncoderBlock(n_embed, n_head=n_head),
            EncoderBlock(n_embed, n_head=n_head),
            EncoderBlock(n_embed, n_head=n_head),
            nn.LayerNorm(n_embed),
        )
        
        # Decoder: processes target language with cross-attention to encoder
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(n_embed, n_head=n_head),
            DecoderBlock(n_embed, n_head=n_head),
            DecoderBlock(n_embed, n_head=n_head),
        ])
        self.decoder_ln = nn.LayerNorm(n_embed)
        
        # Output projection to target vocabulary
        self.lm_head = nn.Linear(n_embed, fr_vocab_size)

    def forward(self, src_idx, tgt_idx, tgt_labels=None):
        """
        src_idx: (B, T_src) - source (English) token indices
        tgt_idx: (B, T_tgt) - target (French) input token indices (shifted for teacher forcing)
        tgt_labels: (B, T_tgt) - target labels for loss computation
        """
        B, T_src = src_idx.shape
        B, T_tgt = tgt_idx.shape
        
        # Encode source
        src_tok_emb = self.src_token_embedding(src_idx)  # (B, T_src, n_embed)
        src_pos_emb = self.src_position_embedding(torch.arange(T_src, device=device))
        src_x = src_tok_emb + src_pos_emb
        encoder_out = self.encoder_blocks(src_x)  # (B, T_src, n_embed)
        
        # Decode target
        tgt_tok_emb = self.tgt_token_embedding(tgt_idx)  # (B, T_tgt, n_embed)
        tgt_pos_emb = self.tgt_position_embedding(torch.arange(T_tgt, device=device))
        decoder_x = tgt_tok_emb + tgt_pos_emb
        
        # Pass through decoder blocks with cross-attention
        for decoder_block in self.decoder_blocks:
            decoder_x = decoder_block(decoder_x, encoder_out)
        decoder_x = self.decoder_ln(decoder_x)
        
        # Project to target vocabulary
        logits = self.lm_head(decoder_x)  # (B, T_tgt, tgt_vocab_size)

        if tgt_labels is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            tgt_labels = tgt_labels.view(B*T)
            # Use label smoothing if specified
            if self.label_smoothing > 0:
                loss = F.cross_entropy(logits, tgt_labels, label_smoothing=self.label_smoothing)
            else:
                loss = F.cross_entropy(logits, tgt_labels)

        return logits, loss

    def generate(self, src_idx, max_new_tokens):
        """
        Generate translation given source sentence
        src_idx: (B, T_src) - source (English) token indices
        """
        B, T_src = src_idx.shape
        
        # Encode source
        src_tok_emb = self.src_token_embedding(src_idx)
        src_pos_emb = self.src_position_embedding(torch.arange(T_src, device=device))
        src_x = src_tok_emb + src_pos_emb
        encoder_out = self.encoder_blocks(src_x)
        
        # Start with SOS token (0)
        tgt_idx = torch.zeros((B, 1), dtype=torch.long, device=device)
        
        for _ in range(max_new_tokens):
            T_tgt = tgt_idx.shape[1]
            
            # Decode current target sequence
            tgt_tok_emb = self.tgt_token_embedding(tgt_idx)
            tgt_pos_emb = self.tgt_position_embedding(torch.arange(T_tgt, device=device))
            decoder_x = tgt_tok_emb + tgt_pos_emb
            
            # Pass through decoder blocks
            for decoder_block in self.decoder_blocks:
                decoder_x = decoder_block(decoder_x, encoder_out)
            decoder_x = self.decoder_ln(decoder_x)
            
            # Get logits for next token
            logits = self.lm_head(decoder_x)  # (B, T_tgt, tgt_vocab_size)
            logits = logits[:, -1, :]  # (B, tgt_vocab_size) - last token
            probs = F.softmax(logits, dim=-1)
            tgt_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            tgt_idx = torch.cat((tgt_idx, tgt_next), dim=1)  # (B, T_tgt+1)
            
            # Stop if all sequences generate EOS (0) - simple stopping condition
            # In practice, you'd use a proper EOS token
        
        return tgt_idx


model = EncoderDecoderModel(label_smoothing=label_smoothing)
model = model.to(device)

# Add weight decay for regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler - reduce LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=500
)

# Early stopping to prevent overfitting
best_val_loss = float('inf')
patience = 1000
patience_counter = 0
best_model_state = None

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        val_loss = losses['val']
        print(f"step {iter} train loss: {losses['train']:.4f}, val loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += eval_interval
            if patience_counter >= patience:
                print(f"Early stopping at iteration {iter} - no improvement for {patience} iterations")
                # Restore best model
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    
    src_batch, tgt_batch, tgt_labels = get_batch('train')
    logits, loss = model(src_batch, tgt_batch, tgt_labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

# Example translation: translate first English sentence
if len(en_sentences) > 0:
    example_en = en_sentences[0]
    example_en_encoded = bpa.encode(example_en, en_merges)[:block_size]
    example_en_encoded = example_en_encoded + [0] * max(0, block_size - len(example_en_encoded))
    src_context = torch.tensor([example_en_encoded], dtype=torch.long, device=device)
    
    print(f"\nSource (English): {example_en}")
    print(f"Expected (French): {fr_sentences[0]}")
    print("\nGenerated translation:")
    generated = model.generate(src_context, max_new_tokens=block_size)
    generated_tokens = generated[0].tolist()
    # Filter out padding tokens (0s) for cleaner output
    generated_tokens = [t for t in generated_tokens if t != 0]
    print(bpa.decode(generated_tokens, fr_merges))
