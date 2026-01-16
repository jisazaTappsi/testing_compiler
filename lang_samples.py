import data
from util import *
from lang_train import CrossAttentionTransformer, block_size, model_name


def get_sample_val_data(num=20):
    rows_in = get_first_rows_fast('OpenSubtitles.en', max_pairs)
    rows_out = get_first_rows_fast('OpenSubtitles.es', max_pairs)

    n = int(train_split_ratio * len(rows_in))

    # Choose samples of validation data
    in_sentences = rows_in[n:]
    out_sentences = rows_out[n:]
    random_val_ids = torch.randint(len(in_sentences), (num,))
    in_sentences = [in_sentences[i] for i in random_val_ids]
    out_sentences = [out_sentences[i] for i in random_val_ids]

    in_merges, out_merges = data.get_merges('lang')
    return data.get_pairs(in_sentences, out_sentences, in_merges, out_merges, block_size, 'lang')


def run():
    # Load data and merges
    val_samples = get_sample_val_data(num=100)
    in_merges, out_merges = data.get_merges('lang')

    model = CrossAttentionTransformer()
    model = model.to(device)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    # Run generation
    tokens = data.get_start_and_end_tokens('lang')
    for row in val_samples:
        context = torch.tensor([[tokens['start_out']]], dtype=torch.long, device=device)
        data_in = row['x_in']
        print(f'I: {data.decode([e for e in data_in if e != tokens['end_in']], in_merges)}')
        translation = data.decode(
            model.generate(x_out=context,
                           x_in=torch.tensor([data_in], dtype=torch.long, device=device),
                           max_new_tokens=block_size)[0].tolist(),
            out_merges
        )
        print(f'O: {translation}\n')


if __name__ == '__main__':
    run()
