import os
import json
from util import *

out_vocab_size = 1000  # Target (Spanish) vocabulary size
in_vocab_size = 1000  # Source (English) vocabulary size - same as out so end tokens match
max_merge_pairs = 10_000


def get_max_merge(my_merge):
    return max(my_merge.values()) if my_merge else 255


def get_start_and_end_tokens_as_list(my_merge):
    merge_max = get_max_merge(my_merge)
    return [merge_max+1, merge_max+2]


def get_start_and_end_tokens():
    in_merges, out_merges = get_merges()
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


def get_merge_filename(input_type):
    return f'merges_{input_type}.json'


def train_merges(toks, input_type, target_vocab_size=280):
    ids = toks.copy()
    idx = 256
    my_merges = {}  # pair => idx
    for i in range(target_vocab_size - idx - 2):  # Saves 2 tokens for START and END
        pair, pair_count = get_max_pair(ids)
        decoded_pair = decode(pair, my_merges)
        print(f'Merging "{decode([pair[0]], my_merges)}", "{decode([pair[1]], my_merges)}" into: "{decoded_pair}"')
        ids = merge(ids, pair, idx=idx)
        my_merges[pair] = idx
        idx += 1

    print(f'tokens len {len(toks)}')
    print(f'ids len: {len(ids)}')
    print(f'compression: {round(len(toks) / len(ids), 2)}')

    # Convert tuple keys to strings for JSON
    with open(get_merge_filename(input_type), 'w') as outfile:
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


def get_merge(input_type):
    with open(get_merge_filename(input_type), 'r') as file:
        data = json.load(file)
        # Convert string keys back to tuples
        return {tuple(map(int, k.split(','))): v for k, v in data.items()}


def get_merges():
    try:
        return get_merge('in'), get_merge('out')
    except FileNotFoundError:
        return save_code_merges()


def load_code_tokens():
    rows = get_first_rows_fast(dataset_name, max_pairs)
    train_rows = get_train_data(rows)[:max_merge_pairs]
    lex_texts, ast_texts = zip(*[(row.split(',')[0], row.split(',')[1]) for row in train_rows])

    lex_text = ' '.join(lex_texts)
    ast_text = ' '.join(ast_texts)
    
    lex_tokens = [int(i) for i in lex_text.encode('utf-8')]
    ast_tokens = [int(i) for i in ast_text.encode('utf-8')]
    print(f'Lex tokens length: {len(lex_tokens)}')
    print(f'AST tokens length: {len(ast_tokens)}')
    return lex_tokens, ast_tokens


def save_code_merges():
    # When recalculating merges needs to delete the model first, as it will lose the encoding :(
    try:
        os.remove(code_model_name)
    except FileNotFoundError:
        pass

    lex_tokens, ast_tokens = load_code_tokens()
    in_merges = train_merges(lex_tokens + ast_tokens, 'in', target_vocab_size=out_vocab_size)
    out_merges = train_merges(lex_tokens + ast_tokens, 'out', target_vocab_size=in_vocab_size)
    return in_merges, out_merges


def get_code_pairs(rows, in_merges, out_merges, block_size):
    pairs = []
    cut_size = block_size - 2  # 1 more token to divide it between x and y upstream on the get_batch method.
    tokens = get_start_and_end_tokens()
    for row in rows:
        lex_sent, ast_sent, result, has_error, idx = row.split(',')
        lex_encoded = encode(lex_sent, in_merges)
        ast_encoded = encode(ast_sent, out_merges)
        # Truncate/pad to block_size
        lex_encoded = lex_encoded[:cut_size]
        lex_encoded = [tokens['start_in']] + lex_encoded + [tokens['end_in']] * max(1, block_size - len(lex_encoded))
        ast_encoded = ast_encoded[:cut_size]
        ast_encoded = [tokens['start_out']] + ast_encoded + [tokens['end_out']] * max(1, block_size - len(ast_encoded))
        pairs.append({'x_in': lex_encoded, 'x_out': ast_encoded, 'has_error': has_error == 'True', 'id': idx})
    return pairs


def get_train_data(iterable):
    n = int(train_split_ratio * len(iterable))
    return iterable[:n]


def get_val_data(iterable):
    n = int(train_split_ratio * len(iterable))
    return iterable[n:]


def get_code_data():
    """We translate from Lex to AST, ie our x_in=LEX, while x_out=AST"""
    rows = get_first_rows_fast(dataset_name, max_pairs)
    in_merges, out_merges = get_merges()
    pairs = get_code_pairs(rows, in_merges, out_merges, block_size)

    train_pairs = get_train_data(pairs)
    val_pairs = get_val_data(pairs)
    print(f'Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}')
    return {
        'train': train_pairs,
        'val': val_pairs
    }, out_merges, in_merges


if __name__ == '__main__':
    save_code_merges()
