import json
import os
from util import *

out_vocab_size = 4_096  # Target (Spanish) vocabulary size
in_vocab_size = 4_096  # Source (English) vocabulary size - same as out so end tokens match
max_pairs = 10_000

def get_max_merge(my_merge):
    return max(my_merge.values()) if my_merge else 255


def get_start_and_end_tokens_as_list(my_merge):
    merge_max = get_max_merge(my_merge)
    return [merge_max+1, merge_max+2]


def get_start_and_end_tokens(model_type):
    in_merges, out_merges = get_merges(model_type)
    merge_max_in = get_max_merge(in_merges)
    merge_max_out = get_max_merge(out_merges)
    return {f'start_in': merge_max_in + 1, f'end_in': merge_max_in + 2,
            f'start_out': merge_max_out + 1, f'end_out': merge_max_out + 2}


def get_max_pair(ids):
    counts = {}
    max_pair = None
    max_pair_count = 0
    for key in zip(ids[:-1], ids[1:]):
        count = counts.get(key, 0) + 1
        counts[key] = count
        if max_pair_count < count:
            max_pair_count = count
            max_pair = key
    return max_pair, max_pair_count

def get_stats(ids):
    counts = {}
    for pair in zip(ids[:-1], ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and pair[0] == ids[i] and pair[1] == ids[i + 1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def get_merge_filename(model_type, input_type):
    return f'{model_type}_merges_{input_type}.json'


def train_merges(toks, model_type, input_type, target_vocab_size=280):
    ids = list(toks)
    idx = 256
    my_merges = {}  # pair => idx
    for i in range(target_vocab_size - idx - 2):  # Saves 2 tokens for START and END
        pair, pair_count = get_max_pair(ids)
        print(f'merging {pair} into a new token {idx}')
        ids = merge(ids, pair, idx=idx)
        my_merges[pair] = idx
        idx += 1

    print(f'tokens len {len(toks)}')
    print(f'ids len: {len(ids)}')
    print(f'compression: {round(len(toks) / len(ids), 2)}')

    # Convert tuple keys to strings for JSON
    with open(get_merge_filename(model_type, input_type), 'w') as outfile:
        json.dump({f'{k[0]},{k[1]}': v for k, v in my_merges.items()}, outfile)

    return my_merges

def decode(ids, my_merges):
    vocab = get_vocab(my_merges)
    tokens = get_start_and_end_tokens_as_list(my_merges)
    toks = b''.join(vocab[i] for i in ids if i not in tokens)
    return toks.decode('utf-8', errors='replace')

def encode(string, my_merges):
    toks = list(string.encode('utf-8'))
    while len(toks) > 1:
        stats = get_stats(toks)
        pair = min(stats, key=lambda p: my_merges.get(p, float('inf')))
        if pair in my_merges:
            toks = merge(toks, pair, idx=my_merges[pair])
        else:
            break
    return toks

def get_vocab(merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx  in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab


def get_merge(model_type, input_type):
    with open(get_merge_filename(model_type, input_type), 'r') as file:
        data = json.load(file)
        # Convert string keys back to tuples
        return {tuple(map(int, k.split(','))): v for k, v in data.items()}


def get_merges(model_type):
    try:
        return get_merge(model_type, 'in'), get_merge(model_type, 'out')
    except FileNotFoundError:
        return save_merges()


def get_last_rows_fast(filename):
    with open(filename, 'rb') as f:  # Open in binary mode
        f.seek(0, os.SEEK_END)
        file_size = f.tell()

        # Estimate how many bytes to read (avg 100 bytes per line * max_pairs)
        # We multiply by 2 or 4 to be safe and ensure we get enough lines
        buffer_size = max_pairs * 200

        if buffer_size > file_size:
            f.seek(0)
        else:
            f.seek(-buffer_size, os.SEEK_END)

        # Read the chunk and decode to text
        chunk = f.read().decode('utf-8', errors='ignore')

        # Get the lines and take the last N
        return chunk.splitlines()[-max_pairs:]


def load_tokens():
    es_sentences = get_last_rows_fast('OpenSubtitles.es')
    es_text = ' '.join(es_sentences)

    en_sentences = get_last_rows_fast('OpenSubtitles.en')
    en_text = ' '.join(en_sentences)

    en_tokens = [int(i) for i in en_text.encode('utf-8')]
    es_tokens = [int(i) for i in es_text.encode('utf-8')]
    print(f'English tokens length: {len(en_tokens)}')
    print(f'Spanish tokens length: {len(es_tokens)}')
    return en_tokens, es_tokens


def save_merges():
    en_tokens, fr_tokens = load_tokens()
    in_merges = train_merges(en_tokens, 'lang', 'in', target_vocab_size=out_vocab_size)
    out_merges = train_merges(fr_tokens, 'lang', 'out', target_vocab_size=in_vocab_size)
    return in_merges, out_merges


def get_pairs(in_sentences, out_sentences, in_merges, out_merges, block_size, model_type):
    pairs = []
    cut_size = block_size - 2  # 1 more token to divide it between x and y upstream on the get_batch method.
    tokens = get_start_and_end_tokens(model_type)
    for en_sent, es_sent in zip(in_sentences, out_sentences):
        en_encoded = encode(en_sent, in_merges)
        es_encoded = encode(es_sent, out_merges)
        # Truncate/pad to block_size
        en_encoded = en_encoded[:cut_size]
        en_encoded = [tokens['start_in']] + en_encoded + [tokens['end_in']] * max(1, block_size - len(en_encoded))
        es_encoded = es_encoded[:cut_size]
        es_encoded = [tokens['start_out']] + es_encoded + [tokens['end_out']] * max(1, block_size - len(es_encoded))
        pairs.append({'x_in': en_encoded, 'x_out': es_encoded})
    return pairs


def get_lang_data():
    """We translate from EN to ES, ie our x_in=EN, while x_out=ES"""
    in_sentences = get_first_rows_fast('OpenSubtitles.en', max_pairs)
    out_sentences = get_first_rows_fast('OpenSubtitles.es', max_pairs)
    in_merges, out_merges = get_merges('lang')
    pairs = get_pairs(in_sentences, out_sentences, in_merges, out_merges, block_size, 'lang')

    n = int(train_split_ratio * len(in_sentences))
    train_pairs = pairs[:n]
    val_pairs = pairs[n:]
    print(f'Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}')
    return {
        'train': train_pairs,
        'val': val_pairs
    }, out_merges, in_merges


if __name__ == '__main__':
    save_merges()
