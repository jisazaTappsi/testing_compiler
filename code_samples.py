import statistics

import pandas as pd

import data
import basic
import tokens
from tokens import *
from util import *
from basic import Parser
from code_train import CrossAttentionTransformer, block_size

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
elif torch.backends.mps.is_available():
    torch.mps.manual_seed(42)


def get_sample_val_data(num):
    df = pd.read_pickle(dataset_name)
    df = df.head(max_samples)

    # Choose samples of validation data
    val_df = data.get_val_data(df)
    random_val_ids = torch.randint(len(val_df), (num,))
    random_val_df = val_df.iloc[random_val_ids]

    return data.get_code_dicts(random_val_df)


def get_hand_made_data():
    rows = []
    hand_samples = ['2456+2235', '3544*4567+6567/7889', '6899/7667', '1899*2908', '1959.14 - 3257', '7758 * 7000 + 6000 ']
    hand_samples += ['2+2', '3*4+6/7', '6/7', '1*2', '1.1 - 3', '7 * 7 + 6 ']

    for idx, hand_sample in enumerate(hand_samples):
        lexer = basic.Lexer('<stdin>', hand_sample)
        token_list, _ = lexer.make_tokens()
        lexer_text = ' '.join(t.__repr__() for t in token_list)

        ast = basic.Parser(token_list).parse()
        interpreter = basic.Interpreter()
        res = interpreter.visit(ast.node, '<program>')

        rows.append(
            {
                'lex_text': lexer_text,
                'ast_text': f'{tokens.SOF} {ast.node} {tokens.EOF}',
                'result': res.value,
                'has_error': False,
                'text': hand_sample,
                'x_in': data.add_pad_tokens_and_trim(data.encode(lexer_text, {}), block_size),
                'x_out': data.add_pad_tokens_and_trim(data.encode(str(ast), {}), block_size),
                'id': idx,
            }
        )

    df = pd.DataFrame(rows)
    return data.get_code_dicts(df)


def sample_decode(my_data, merges):
    return CrossAttentionTransformer.fix_unmatched_parenthesis(data.decode(my_data, merges))


def run(num_samples):

    # Load data and merges
    val_samples = get_sample_val_data(num=num_samples)
    val_samples += get_hand_made_data()
    lex_merges, ast_merges = data.get_merges()

    model = CrossAttentionTransformer()
    model = model.to(device)
    model.load_state_dict(torch.load(code_model_name))
    model.eval()

    # Run generation
    tree_scores = []
    computation_scores = []
    for row in val_samples:
        data_in = row['x_in']
        data_out = row['x_out']
        has_error = row['has_error']
        print(f'text: {row['text']}')
        predicted_ast_text = model.inference(data_in, ast_merges)
        print(f'P(#{predicted_ast_text.count('(') - predicted_ast_text.count(')')}): {predicted_ast_text}')
        target_ast_text = sample_decode(data_out, ast_merges)

        print(f"T(#{target_ast_text.count('(') - target_ast_text.count(')')}): {target_ast_text}")
        equal = predicted_ast_text == target_ast_text
        tree_scores.append(int(equal))

        try:
            predicted_ast = Parser.get_tree_from_string(predicted_ast_text)
            predicted_res = basic.Interpreter().visit(predicted_ast, '<program>')
            if predicted_res.error:
                print(f'predicted_ast interpretation error: {predicted_res.error}\n')
                computation_scores.append(0)
                continue
        except Exception as e:
            print(f'building/executing predicted AST gets: {e}, continuing...\n')
            computation_scores.append(0)
            continue

        try:
            target_ast = Parser.get_tree_from_string(target_ast_text)
            target_res = basic.Interpreter().visit(target_ast, '<program>')
        except Exception as e:
            # If the algorithm is cutting the data, then the algorithm is at fault... :(
            print(f'building/executing target AST gets: {e}, continuing...\n')
            computation_scores.append(0)
            continue

        try:
            is_close = torch.allclose(torch.tensor(float(predicted_res.value.value)),
                                      torch.tensor(float(target_res.value.value)))
        except Exception as e:
            print(f'cannot compare target and predicted results: {e}')
            is_close = False

        computation_scores.append(int(is_close))
        print(f'{has_error=} | AST are equal: {equal} | predicted: {predicted_res.value} target: {target_res.value} | computation is equal: {is_close}\n')

    print(f'Avg performance tree_scores: {round(statistics.mean(tree_scores)*100)}%')
    computation_percentage = statistics.mean(computation_scores) * 100
    print(f'Avg performance computation: {round(computation_percentage)}%')
    return computation_percentage


if __name__ == '__main__':
    run(num_samples=250)
