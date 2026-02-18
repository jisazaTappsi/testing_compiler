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

    # Choose samples of validation data
    val_df = data.get_val_data(df)
    random_val_ids = torch.randint(len(val_df), (num,))
    random_val_df = val_df.iloc[random_val_ids]

    return random_val_df


def get_hand_made_data():
    rows = []
    hand_samples = ['2456+2235', '3544*4567+6567/7889', '6899/7667', '1899*2908', '1959.14 - 3257', '7758 * 7000 + 6000 ']
    hand_samples += ['2+2', '3*4+6/7', '6/7', '1*2', '1.1 - 3', '7 * 7 + 6 ']
    #hand_samples = ['6.12++(+(412.41/2)/+(4*1420))']

    for idx, hand_sample in enumerate(hand_samples):
        lexer = basic.Lexer('<stdin>', hand_sample)
        token_list, _ = lexer.make_tokens()
        lexer_text = ' '.join(t.__repr__() for t in token_list)

        ast = basic.Parser(token_list).parse()
        interpreter = basic.Interpreter()
        ctx = basic.Context('<program>')
        ctx.symbol_table = basic.get_symbol_table()
        res = interpreter.visit(ast.node, ctx)

        rows.append(
            {
                'lex_text': lexer_text,
                'ast_text': f'{tokens.SOF} {ast.node} {tokens.EOF}',
                'result': res.value,
                'text': hand_sample,
                'x_in': data.add_pad_tokens_and_trim(data.encode(lexer_text, {}), block_size),
                'x_out': data.add_pad_tokens_and_trim(data.encode(f'{tokens.SOF} {ast.node} {tokens.EOF}', {}), block_size),
                'id': idx,
            }
        )

    return pd.DataFrame(rows)


def sample_decode(my_data, merges):
    return CrossAttentionTransformer.fix_unmatched_parenthesis(data.decode(my_data, merges))


def symbol_tables_equal(pred_symbols, target_symbols):
    """Compare two symbol-table dicts (name -> basic.Number). Same keys and same numeric values."""
    if set(pred_symbols.keys()) != set(target_symbols.keys()):
        return False
    for name in pred_symbols:
        p, t = pred_symbols[name], target_symbols[name]
        if type(p) != type(t):
            return False
        if hasattr(p, 'value'):  # basic.Number
            try:
                if not torch.allclose(torch.tensor(float(p.value)), torch.tensor(float(t.value))):
                    return False
            except (TypeError, ValueError):
                if p.value != t.value:
                    return False
        else:
            if p != t:
                return False
    return True


def run(num_samples):

    # Load data and merges
    val_samples = get_sample_val_data(num=num_samples)
    #val_samples = get_hand_made_data()
    lex_merges, ast_merges = data.get_merges()

    model = CrossAttentionTransformer()
    model = model.to(device)
    model.load_state_dict(torch.load(code_model_name))
    model.eval()

    tree_scores_per_sample = []
    computation_scores_per_sample = []
    for idx, sample in val_samples.iterrows():
        predicted_ctx = basic.Context('<program>')
        predicted_ctx.symbol_table = basic.get_symbol_table()
        target_ctx = basic.Context('<program>')
        target_ctx.symbol_table = basic.get_symbol_table()
        sample_run_ok = True
        statement_ast_matches = []

        print(f'{idx}:\n{sample.lexer_text}')
        for x_in, x_out in zip(sample.x_in, sample.x_out):
            predicted_ast_text = model.inference(x_in, ast_merges)
            print(f'P(#{predicted_ast_text.count("(") - predicted_ast_text.count(")")}): {predicted_ast_text}')
            target_ast_text = sample_decode(x_out, ast_merges)
            print(f"T(#{target_ast_text.count('(') - target_ast_text.count(')')}): {target_ast_text}\n")

            statement_ast_matches.append(predicted_ast_text == target_ast_text)

            try:
                predicted_ast = Parser.get_tree_from_string(predicted_ast_text)
                predicted_res = basic.Interpreter().visit(predicted_ast, predicted_ctx)
                if predicted_res.error:
                    sample_run_ok = False
                    break
            except Exception as e:
                print(f'building/executing predicted AST gets: {e}, continuing...\n')
                sample_run_ok = False
                break

            try:
                target_ast = Parser.get_tree_from_string(target_ast_text)
                target_res = basic.Interpreter().visit(target_ast, target_ctx)
                if target_res.error:
                    sample_run_ok = False
                    break
            except Exception as e:
                print(f'building/executing target AST gets: {e}, continuing...\n')
                sample_run_ok = False
                break

        # Per-sample tree: 1 iff every statement had correct AST AND the sample ran
        # (this way, any parse/interpret error forces tree=0, and if tree=1 we know
        # both predicted and target programs were actually executable).
        tree_ok = sample_run_ok and all(statement_ast_matches)
        tree_scores_per_sample.append(int(tree_ok))

        if sample_run_ok:
            tables_match = symbol_tables_equal(predicted_ctx.symbol_table.symbols, target_ctx.symbol_table.symbols)
            computation_scores_per_sample.append(int(tables_match))
        else:
            tables_match = False
            computation_scores_per_sample.append(0)

        print(f'Trees are equal: {tree_ok} | predicted: {predicted_ctx.symbol_table.symbols} target: {target_ctx.symbol_table.symbols} | computation is equal: {tables_match}\n')

    n_samples = len(computation_scores_per_sample)
    tree_pct = statistics.mean(tree_scores_per_sample) * 100 if tree_scores_per_sample else 0
    computation_pct = statistics.mean(computation_scores_per_sample) * 100 if computation_scores_per_sample else 0
    print(f'Per-sample (n={n_samples}): Tree {round(tree_pct, 3)}%  |  Computation {round(computation_pct, 3)}%')
    return computation_pct


if __name__ == '__main__':
    run(num_samples=250)
