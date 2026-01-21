import statistics

import data
import basic
from basic import Parser
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
    tree_scores = []
    computation_scores = []
    for row in val_samples:
        context = torch.tensor([[tokens['start_out']]], dtype=torch.long, device=device)
        data_in = row['x_in']
        print(f'I: {sample_decode(data_in, tokens, in_merges)}')
        predicted_ast_text = data.decode(
            model.generate(x_out=context,
                           x_in=torch.tensor([data_in], dtype=torch.long, device=device),
                           max_new_tokens=block_size)[0].tolist(),
            out_merges
        )
        print(f'O(#{predicted_ast_text.count('(') - predicted_ast_text.count(')')}): {predicted_ast_text}')
        target_ast_text = sample_decode(row['x_out'], tokens, out_merges)
        print(f"T(#{target_ast_text.count('(') - target_ast_text.count(')')}): {target_ast_text}")
        equal = predicted_ast_text == target_ast_text
        print(f'AST are equal: {equal}\n')
        tree_scores.append(int(equal))

        try:
            predicted_ast = Parser.get_tree_from_string(predicted_ast_text)
            predicted_res = basic.Interpreter().visit(predicted_ast)
            if predicted_res.error:
                print(f'predicted_ast interpretation error: {predicted_res.error}')
                computation_scores.append(0)
                continue
            else:
                print(f'predicted value: {predicted_res.value}')
        except Exception as e:
            print(f'building/executing predicted AST gets: {e}, continuing...')
            computation_scores.append(0)
            continue

        target_ast = Parser.get_tree_from_string(target_ast_text)
        target_res = basic.Interpreter().visit(target_ast)
        print(target_res.value)

        is_close = torch.allclose(torch.tensor(float(predicted_res.value.value)), torch.tensor(float(target_res.value.value)))
        computation_scores.append(int(is_close))
        print(f'computation is equal: {is_close}')

    print(f'Avg performance tree_scores: {round(statistics.mean(tree_scores)*100)}%')
    print(f'Avg performance computation: {round(statistics.mean(computation_scores)*100)}%')

if __name__ == '__main__':
    run()
