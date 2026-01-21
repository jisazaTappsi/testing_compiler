import csv
import json
import os
from util import *

out_vocab_size = 1000  # Target (Spanish) vocabulary size
in_vocab_size = 1000  # Source (English) vocabulary size - same as out so end tokens match
max_merge_pairs = 10_000

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
        decoded_pair = decode(pair, my_merges)
        print(f'merging "{decode([pair[0]], my_merges)}" and "{decode([pair[1]], my_merges)}" into a new token "{decoded_pair}"')
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
        if model_type == 'lang':
            return save_lang_merges()
        elif model_type == 'code':
            return save_code_merges()


def get_last_rows_fast(filename):
    with open(filename, 'rb') as f:  # Open in binary mode
        f.seek(0, os.SEEK_END)
        file_size = f.tell()

        # Estimate how many bytes to read (avg 100 bytes per line * max_merge_pairs)
        # We multiply by 2 or 4 to be safe and ensure we get enough lines
        buffer_size = max_merge_pairs * 200

        if buffer_size > file_size:
            f.seek(0)
        else:
            f.seek(-buffer_size, os.SEEK_END)

        # Read the chunk and decode to text
        chunk = f.read().decode('utf-8', errors='ignore')

        # Get the lines and take the last N
        return chunk.splitlines()[-max_merge_pairs:]


def load_lang_tokens():
    es_sentences = get_last_rows_fast('OpenSubtitles.es')
    es_text = ' '.join(es_sentences)

    en_sentences = get_last_rows_fast('OpenSubtitles.en')
    en_text = ' '.join(en_sentences)

    en_tokens = [int(i) for i in en_text.encode('utf-8')]
    es_tokens = [int(i) for i in es_text.encode('utf-8')]
    print(f'English tokens length: {len(en_tokens)}')
    print(f'Spanish tokens length: {len(es_tokens)}')
    return en_tokens, es_tokens


def save_lang_merges():
    # When recalculating merges needs to delete the model first, as it will lose the encoding :(
    try:
        os.remove(lang_model_name)
    except FileNotFoundError:
        pass

    en_tokens, es_tokens = load_lang_tokens()
    in_merges = train_merges(en_tokens, 'lang', 'in', target_vocab_size=out_vocab_size)
    out_merges = train_merges(es_tokens, 'lang', 'out', target_vocab_size=in_vocab_size)
    return in_merges, out_merges


def load_code_tokens():
    with open('dataset.csv', 'r') as f:
        reader = csv.reader(f)
        lex_texts, ast_texts = zip(*[(row[0], row[1]) for idx, row in enumerate(reader) if idx < max_merge_pairs])
    
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
    in_merges = train_merges(lex_tokens, 'code', 'in', target_vocab_size=out_vocab_size)
    out_merges = train_merges(ast_tokens, 'code', 'out', target_vocab_size=in_vocab_size)
    return in_merges, out_merges


def get_lang_pairs(in_sentences, out_sentences, in_merges, out_merges, block_size, model_type):
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


def get_code_pairs(rows, in_merges, out_merges, block_size, model_type):
    pairs = []
    cut_size = block_size - 2  # 1 more token to divide it between x and y upstream on the get_batch method.
    tokens = get_start_and_end_tokens(model_type)
    for row in rows:
        lex_sent, ast_sent, _ = row.split(',')
        lex_encoded = encode(lex_sent, in_merges)
        ast_encoded = encode(ast_sent, out_merges)
        # Truncate/pad to block_size
        lex_encoded = lex_encoded[:cut_size]
        lex_encoded = [tokens['start_in']] + lex_encoded + [tokens['end_in']] * max(1, block_size - len(lex_encoded))
        ast_encoded = ast_encoded[:cut_size]
        ast_encoded = [tokens['start_out']] + ast_encoded + [tokens['end_out']] * max(1, block_size - len(ast_encoded))
        pairs.append({'x_in': lex_encoded, 'x_out': ast_encoded})
    return pairs


def get_lang_data():
    """We translate from EN to ES, ie our x_in=EN, while x_out=ES"""
    in_sentences = get_first_rows_fast('OpenSubtitles.en', max_pairs)
    out_sentences = get_first_rows_fast('OpenSubtitles.es', max_pairs)
    in_merges, out_merges = get_merges('lang')
    pairs = get_lang_pairs(in_sentences, out_sentences, in_merges, out_merges, block_size, 'lang')

    n = int(train_split_ratio * len(in_sentences))
    train_pairs = pairs[:n]
    val_pairs = pairs[n:]
    print(f'Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}')
    return {
        'train': train_pairs,
        'val': val_pairs
    }, out_merges, in_merges


def get_code_data():
    """We translate from Lex to AST, ie our x_in=LEX, while x_out=AST"""
    rows = get_first_rows_fast('dataset.csv', max_pairs)
    in_merges, out_merges = get_merges('code')
    pairs = get_code_pairs(rows, in_merges, out_merges, block_size, 'code')

    n = int(train_split_ratio * len(pairs))
    train_pairs = pairs[:n]
    val_pairs = pairs[n:]
    print(f'Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}')
    return {
        'train': train_pairs,
        'val': val_pairs
    }, out_merges, in_merges


if __name__ == '__main__':
    save_lang_merges()
