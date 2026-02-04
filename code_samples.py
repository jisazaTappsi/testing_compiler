import statistics

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
    rows = get_first_rows_fast(dataset_name, max_pairs)

    # Choose samples of validation data
    val_rows = data.get_val_data(rows)
    random_val_ids = torch.randint(len(val_rows), (num,))
    random_val_rows = [val_rows[i] for i in random_val_ids]

    lex_merges, ast_merges = data.get_merges()
    pairs = data.get_code_pairs(random_val_rows, lex_merges, ast_merges, block_size)
    if introduce_error:
        return [e for e in pairs if e['has_error']]
    else:
        return pairs


def get_hand_made_data():
    rows = []
    #hand_samples = ['2456+2235', '3544*4567+6567/7889', '6899/7667', '1899*2908', '195.14 - 3257', '7758 * 7000 + 6 ']
    hand_samples = ['2+2', '3*4+6/7', '6/7', '1*2', '1.1 - 3', '7 * 7 + 6 ']

    for idx, hand_sample in enumerate(hand_samples):
        lexer = basic.Lexer('<stdin>', hand_sample)
        token_list, _ = lexer.make_tokens()
        lexer_text = ' '.join(t.__repr__() for t in token_list)

        ast = basic.Parser(token_list).parse()

        interpreter = basic.Interpreter()
        res = interpreter.visit(ast.node, '<program>')

        rows.append(','.join([lexer_text, f'{tokens.TT_SOF} {ast.node} {tokens.TT_EOF}', str(res.value), 'False', str(idx)]))

    lex_merges, ast_merges = data.get_merges()
    pairs = data.get_code_pairs(rows, lex_merges, ast_merges, block_size)
    return pairs


def sample_decode(my_data, merges):
    return data.decode(my_data, merges)


def run(num_samples):

    # Load data and merges
    val_samples = get_sample_val_data(num=num_samples)
    #val_samples = get_hand_made_data()
    lex_merges, ast_merges = data.get_merges()

    model = CrossAttentionTransformer()
    model = model.to(device)
    model.load_state_dict(torch.load(code_model_name))
    model.eval()

    # Run generation
    tree_scores = []
    computation_scores = []
    for row in val_samples:
        context = torch.tensor([[TOKEN_IDS[TT_SOF]]], dtype=torch.long, device=device)
        data_in = row['x_in']
        data_out = row['x_out']
        has_error = row['has_error']
        print(f'I: {sample_decode(data_in, lex_merges)}')
        predicted_ast_text = data.decode(
            model.generate(x_out=context,
                           x_in=torch.tensor([data_in], dtype=torch.long, device=device),
                           max_new_tokens=block_size)[0].tolist(),
            ast_merges
        )
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
    run(num_samples=500)
