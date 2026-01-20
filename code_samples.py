import data
from util import *
from code_train import CrossAttentionTransformer, block_size


def get_sample_val_data(num=20):
    rows = get_first_rows_fast('dataset.csv', max_pairs)

    n = int(train_split_ratio * len(rows))

    # Choose samples of validation data
    val_rows = rows[n:]
    random_val_ids = torch.randint(len(val_rows), (num,))
    random_val_rows = [val_rows[i] for i in random_val_ids]

    in_merges, out_merges = data.get_merges('code')
    return data.get_code_pairs(random_val_rows, in_merges, out_merges, block_size, 'code')


def sample_decode(my_data, tokens, merges):
    return data.decode([e for e in my_data if e != tokens['end_in']], merges)


def run():
    # Load data and merges
    val_samples = get_sample_val_data(num=100)
    in_merges, out_merges = data.get_merges('code')

    model = CrossAttentionTransformer()
    model = model.to(device)
    model.load_state_dict(torch.load(code_model_name))
    model.eval()

    # Run generation
    tokens = data.get_start_and_end_tokens('code')
    for row in val_samples:
        context = torch.tensor([[tokens['start_out']]], dtype=torch.long, device=device)
        data_in = row['x_in']
        print(f'I: {sample_decode(data_in, tokens, in_merges)}')
        translation = data.decode(
            model.generate(x_out=context,
                           x_in=torch.tensor([data_in], dtype=torch.long, device=device),
                           max_new_tokens=block_size)[0].tolist(),
            out_merges
        )
        print(f'O: {translation}')
        print(f'T: {sample_decode(row['x_out'], tokens, out_merges)}\n')


if __name__ == '__main__':
    run()
