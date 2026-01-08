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

def train_merges(toks, target_vocab_size=280):
    ids = list(toks)
    idx = 256
    my_merges = {}  # pair => idx
    for i in range(target_vocab_size - idx):
        pair, pair_count = get_max_pair(ids)
        print(f'merging {pair} into a new token {idx}')
        ids = merge(ids, pair, idx=idx)
        my_merges[pair] = idx
        idx += 1

    print(f'tokens len {len(toks)}')
    print(f'ids len: {len(ids)}')
    print(f'compression: {round(len(toks) / len(ids), 2)}')
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
