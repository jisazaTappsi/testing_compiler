import csv
import json

out_vocab_size = 2_000  # Source (English) vocabulary size
in_vocab_size = 2_000  # Target (French) vocabulary size
max_pairs = 10_000


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

def train_merges(toks, input_type, target_vocab_size=280):
    ids = list(toks)
    idx = 256
    my_merges = {}  # pair => idx
    for i in range(target_vocab_size - idx - 1):  # Saves 1 token for EOF
        pair, pair_count = get_max_pair(ids)
        print(f'merging {pair} into a new token {idx}')
        ids = merge(ids, pair, idx=idx)
        my_merges[pair] = idx
        idx += 1

    print(f'tokens len {len(toks)}')
    print(f'ids len: {len(ids)}')
    print(f'compression: {round(len(toks) / len(ids), 2)}')

    # Convert tuple keys to strings for JSON
    with open(f'merges_{input_type}.json', 'w') as outfile:
        json.dump({f'{k[0]},{k[1]}': v for k, v in my_merges.items()}, outfile)

    return my_merges

def decode(ids, my_merges):
    vocab = get_vocab(my_merges)
    toks = b''.join(vocab[i] for i in ids)
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
    with open(f'merges_{input_type}.json', 'r') as file:
        data = json.load(file)
        # Convert string keys back to tuples
        return {tuple(map(int, k.split(','))): v for k, v in data.items()}


def get_merges():
    try:
        return get_merge('in'), get_merge('out')
    except FileNotFoundError:
        return save_merges()


def load_tokens():
    with open('en_fr_translation.csv', mode='r', newline='') as csv_file:
        translation_corpus = csv.reader(csv_file, delimiter=',')
        rows = list(translation_corpus)[-max_pairs:]
        en_sentences = [row[0] for row in rows]
        fr_sentences = [row[1] for row in rows]
        en_text = ' '.join(en_sentences)
        fr_text = ' '.join(fr_sentences)

    en_tokens = [int(i) for i in en_text.encode('utf-8')]
    fr_tokens = [int(i) for i in fr_text.encode('utf-8')]
    print(f'English tokens length: {len(en_tokens)}')
    print(f'French tokens length: {len(fr_tokens)}')
    return en_tokens, fr_tokens


def save_merges():
    en_tokens, fr_tokens = load_tokens()
    in_merges = train_merges(en_tokens, 'in', target_vocab_size=out_vocab_size)
    out_merges = train_merges(fr_tokens, 'out', target_vocab_size=in_vocab_size)
    return in_merges, out_merges


if __name__ == '__main__':
    save_merges()
