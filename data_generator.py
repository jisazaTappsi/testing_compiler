import math
import re
import random
from typing import Optional

import pandas as pd

import data
import basic
import tokens
from util import *

num_samples = 100_000  # 5_000_000


def log_scale_int(low: int, high: int, base: float = math.e) -> int:
    """Sample an integer in [low, high] with log scale so smaller numbers are overrepresented.
    low must be >= 1 (use separate logic to include 0)."""
    if low >= high:
        return low
    # u uniform in [0,1] -> value log-uniform in [low, high], then round
    u = random.random()
    log_val = math.log(low, base) + u * (math.log(high, base) - math.log(low, base))
    return min(high, max(low, int(round(math.exp(log_val)))))


def int_part():
    # TODO: tests proposed
    #return 0 if random.random() < 0.1 else log_scale_int(1, 2**31, base=10)
    return 0 if random.random() < 0.1 else log_scale_int(1, 9999)


def generate_number():
    """Generate a random integer or float (log scale: smaller numbers more likely)."""
    # Include 0 with small probability; otherwise log-scale in [1, 9999]
    if random.random() < 0.3:  # 30% chance of float
        integer_part = int_part()
        decimal_part = random.randint(0, 99)  # 2 digits id drawn from uniform distribution
        return f"{integer_part}.{decimal_part:02d}"
    else:
        return str(int_part())


def generate_factor(depth=0, max_depth=5, allowed_vars=None):
    """
    Generate a factor according to grammar:
    factor : INT|FLOAT|IDENTIFIER
          : (PLUS|MINUS) factor
          : LPAREN expr RPAREN
    """
    if depth >= max_depth:
        # At max depth, only generate numbers (or variable if allowed) to avoid infinite recursion
        if allowed_vars and random.random() < 0.3:
            return random.choice(allowed_vars)
        return generate_number()
    
    choice = random.random()
    
    if allowed_vars and choice < 0.25:  # 25% chance: use a declared variable
        return random.choice(allowed_vars)
    elif choice < 0.5:  # number (or variable if not using vars)
        if allowed_vars and random.random() < 0.4:
            return random.choice(allowed_vars)
        return generate_number()
    elif choice < 0.75:  # unary operator
        op = random.choice(['+', '-'])
        factor = generate_factor(depth + 1, max_depth, allowed_vars)
        return f"{op}{factor}"
    else:  # parentheses
        expr = generate_expr(depth + 1, max_depth, allowed_vars)
        return f"({expr})"


def generate_term(depth=0, max_depth=5, allowed_vars=None):
    """
    Generate a term according to grammar:
    term : factor ((MUL|DIV) factor)*
    """
    if depth >= max_depth:
        return generate_factor(depth, max_depth, allowed_vars)

    result = generate_factor(depth, max_depth, allowed_vars)
    
    # Randomly add more factors with MUL or DIV operators (often 0 to keep equations small)
    num_ops = 1 if random.random() < 0.3 else 0
    for _ in range(num_ops):

        # Desperate attempt at trying to decrease the division by zeroes...
        op = random.choices(['*', '/'], weights=[0.9, 0.1], k=1)[0]

        while True:  # Don't allow division by zero
            factor = generate_factor(depth, max_depth, allowed_vars)
            try:
                factor = factor.replace('None', '0')  # in our interpreter None is a 0
                mini_expr = int(eval(factor))
                if op != '/' or mini_expr != 0:
                    break
            except:
                break

        result = f"{result}{op}{factor}"
    
    return result


def generate_expr(depth=0, max_depth=5, allowed_vars=None):
    """
    Generate an expression according to grammar:
    expr : (term(PLUS|MINUS) term)*
    """
    if allowed_vars is None:
        allowed_vars = []
    if depth >= max_depth:  # At max depth, generate a simple term
        return generate_term(depth, max_depth, allowed_vars)
    
    result = generate_term(depth, max_depth, allowed_vars)
    
    # Randomly add more terms with PLUS or MINUS operators (often 0 to keep equations small)
    num_ops = 1 if random.random() < 0.25 else 0
    for _ in range(num_ops):
        op = random.choice(['+', '-'])
        term = generate_term(depth, max_depth, allowed_vars)
        result = f"{result}{op}{term}"
    
    return result


def generate_arithmetic_expression(use_variables=False, allowed_vars=None):
    """
    Generate a valid arithmetic expression that can be parsed.
    Optionally limit the length of the generated expression.
    When use_variables is True, factors may be identifiers from allowed_vars (list of variable names).
    """
    max_length = block_size
    vars_list = list(allowed_vars) if use_variables and allowed_vars else []

    # Start with a reasonable max_depth based on block_size. Deeper expressions tend to be longer.
    max_depth = 2
    expr = generate_expr(depth=0, max_depth=max_depth, allowed_vars=vars_list)

    # If expression is too long, regenerate with lower max_depth (stricter cap leaves headroom for comparisons/logic encoding)
    while len(expr) > max_length//10:
        max_depth = max(1, max_depth - 1)
        expr = generate_expr(depth=0, max_depth=max_depth, allowed_vars=vars_list)

    return expr


# Variable names: letters only, no digits. Exclude names that are keywords.
_VAR_NAME_LETTERS = 'abcdefghijklmnopqrstuvwxyz'  # single letters (no 'v' to avoid 'var')


def _new_var_name(declared: list) -> str:
    """Return a variable name not in declared and not a keyword. Letters only."""
    forbidden = set(declared) | set(tokens.KEYWORDS)
    # Single-letter names first
    available = [c for c in _VAR_NAME_LETTERS if c not in forbidden]
    if available:
        return random.choice(available)
    # Fallback: two-letter names (letters only)
    for c1 in _VAR_NAME_LETTERS:
        for c2 in _VAR_NAME_LETTERS:
            name = c1 + c2
            if name not in forbidden:
                return name
    # Last resort: longer names (should not happen in practice)
    for length in range(3, 10):
        for _ in range(100):
            name = ''.join(random.choice(_VAR_NAME_LETTERS) for _ in range(length))
            if name not in forbidden:
                return name
    return 'x'


BOOLEAN_LITERALS = [tokens.NULL, tokens.TRUE, tokens.FALSE]


def generate_program_expression(allowed_vars) -> str:
    """
    Generate an expression compatible with the full grammar, including
    arithmetic, comparison operators, logical AND/OR, and the identifiers
    True, False and null.
    """

    idents = list(allowed_vars) + BOOLEAN_LITERALS
    use_vars = bool(idents)

    def gen_arith():
        return generate_arithmetic_expression(use_variables=use_vars, allowed_vars=idents)

    # Start from either a boolean-like identifier or a plain arithmetic expression
    if random.random() < 0.3:
        expr = random.choice(BOOLEAN_LITERALS)
    else:
        expr = gen_arith()

    # Optionally add one or more comparison operations (slightly lower so more fit in block_size)
    if random.random() < 0.20:
        num_comparisons = random.randint(0, 1)
        for _ in range(num_comparisons):
            op = random.choice(['==', '!=', '<', '>', '<=', '>='])
            right = gen_arith()
            expr = f"{expr}{op}{right}"

    # Optional leading NOT (low prob to keep equations small)
    if random.random() < 0.15:
        expr = f"{tokens.NOT} {expr}"

    # Optionally chain with AND/OR (often 0 to keep equations small)
    num_logic_ops = 1 if random.random() < 0.2 else 0
    for _ in range(num_logic_ops):
        op = random.choice([tokens.AND, tokens.OR])
        right = gen_arith()
        # Sometimes turn the right-hand side into a comparison expression as well
        if random.random() < 0.7:
            num_cmp = random.randint(0, 1)
            for _ in range(num_cmp):
                cmp_op = random.choice(['==', '!=', '<', '>', '<=', '>='])
                right = f"{right}{cmp_op}{gen_arith()}"
        if random.random() < 0.3:
            right = f"{tokens.NOT} {right}"
        expr = f"{expr} {op} {right}"

    return expr


def generate_program_statements(texts) -> list:
    """Generates a short program with valid statements. Each statement is either a variable declaration
    ('var x = expr') or a standalone expression. Expressions may use previously declared variables."""
    declared = []
    statements = []
    num_statements = random.randint(2, 5)

    for _ in range(num_statements):
        while True:
            if not declared or random.random() < 0.6:
                # Variable declaration: var name = expr
                name = _new_var_name(declared)
                expr = generate_program_expression(declared)
                text = f"{tokens.VAR} {name} = {expr}"
                if text not in texts:
                    declared.append(name)
                    break
            else:
                # Standalone expression (can use declared variables)
                text = generate_program_expression(declared)
                if text not in texts:
                    break

        texts.add(text)
        statements.append(text)

    return statements


# ---------------------------------------------------------------------------
# Function call -> body AST samples
# ---------------------------------------------------------------------------
FUNC_TEMPLATES = [
    ("sum", ["a", "b"], "a+b"),
    ("sub", ["a", "b"], "a-b"),
    ("mul", ["a", "b"], "a*b"),
    ("add", ["x", "y"], "x+y"),
    ("diff", ["x", "y"], "x-y"),
    ("prod", ["x", "y"], "x*y"),
    ("double", ["x"], "x+x"),
    ("square", ["x"], "x*x"),
    ("negate", ["x"], "-x"),
    ("triple", ["n"], "n+n+n"),
    ("add3", ["a", "b", "c"], "a+b+c"),
    ("mul3", ["a", "b", "c"], "a*b*c"),
    ("avg", ["a", "b"], "(a+b)/2"),
    ("dist", ["a", "b"], "a-b"),
    ("sumsq", ["a", "b"], "a*a+b*b"),
    ("combo", ["a", "b"], "a*b+a+b"),
    ("scale", ["x", "f"], "x*f"),
    ("halve", ["x"], "x/2"),
    ("inc", ["n"], "n+1"),
    ("dec", ["n"], "n-1"),
]


def _substitute_params(body, params, args):
    """Replace parameter names in body with concrete argument values (whole-word)."""
    result = body
    for param, arg in zip(params, args):
        result = re.sub(rf'\b{re.escape(param)}\b', arg, result)
    return result


def generate_func_call_sample(idx):
    """Generate a single (call_lex, body_ast) pair with random args.
    Returns (lex_text, ast_text, x_in, x_out) or Nones on failure."""
    name, params, body = random.choice(FUNC_TEMPLATES)
    args = [generate_number() for _ in params]

    call_text = f"{name}({', '.join(args)})"
    body_text = _substitute_params(body, params, args)

    # Lex the call (model input)
    lexer_call = basic.Lexer('<stdin>', call_text)
    call_tokens, error = lexer_call.make_tokens()
    if error:
        return None
    lex_text = ' '.join(t.__repr__() for t in call_tokens)

    # Parse the substituted body (model target)
    lexer_body = basic.Lexer('<stdin>', body_text)
    body_tokens, error = lexer_body.make_tokens()
    if error:
        return None

    parser = basic.Parser(body_tokens)
    ast = parser.parse()
    if ast.error:
        return None
    ast_text = f'{tokens.SOF} {ast.node} {tokens.EOF}'

    lex_encoded = data.encode(lex_text, {})
    ast_encoded = data.encode(ast_text, {})
    if len(lex_encoded) > block_size or len(ast_encoded) > block_size:
        return None

    x_in = data.add_pad_tokens_and_trim(lex_encoded, block_size)
    x_out = data.add_pad_tokens_and_trim(ast_encoded, block_size)
    return {
        'lexer_text': f'\n{lex_text}',
        'ast_text': f'\n{ast_text}',
        'text': call_text,
        'x_in': [x_in],
        'x_out': [x_out],
        'symbols': {'_output_list': []},
        'id': idx,
    }


class Sample:
    lexer_text: str
    ast_text: str
    text: str
    x_in: list
    x_out: list
    symbols: dict
    id: Optional[int]

    def __init__(self, statements, idx):
        self.lexer_text = ''
        self.ast_text = ''
        self.text = '\n'.join(statements)
        self.x_in = []
        self.x_out = []
        self.symbols = {'_output_list': []}
        self.id = idx

def print_program(statements):
    print('\n\nProgram sample:')
    print(f'\n'.join(statements))
    print('----------------------------------\n\n')


def generate():
    invalid_count = 0
    rows = []
    texts = set()

    func_call_count = 0

    for idx in range(num_samples):
        """
        if random.random() < FUNC_CALL_RATIO:
            row = generate_func_call_sample(idx)
            if row is None:
                invalid_count += 1
                continue
            rows.append(row)
            func_call_count += 1
            continue
        """

        is_valid = True
        statements = generate_program_statements(texts)
        symbol_table = basic.get_symbol_table()
        sample = Sample(statements, idx)

        if idx % 1_000 == 0:
            print_program(statements)
            print(f"loaded: {(idx/num_samples)*100:.2f}%")

        for text in statements:

            # Verify it can be lexed and parsed
            lexer = basic.Lexer('<stdin>', text)
            try:
                token_list, error = lexer.make_tokens()
                if error:
                    print(f'Lexing is invalid!: {error.as_string()}')
                    invalid_count += 1
                    is_valid = False
                    break
                lexer_text = ' '.join(t.__repr__() for t in token_list)
                sample.lexer_text += f"\n{lexer_text}"

                if random.random() < 0.5:
                    text_error = text.replace('var ', '')
                    lexer_error = basic.Lexer('<stdin>', text_error)
                    token_list_error, error = lexer_error.make_tokens()
                    if error:
                        print(f'Lexing is invalid: {error.as_string()}')
                        invalid_count += 1
                        is_valid = False
                        break
                    lexer_text = ' '.join(t.__repr__() for t in token_list_error)

                # Try to parse
                parser = basic.Parser(token_list)
                ast = parser.parse()
                if ast.error:
                    print(f'Parsing is invalid: {ast.error.as_string()}')
                    invalid_count += 1
                    is_valid = False
                    break
                ast_text = f'{tokens.SOF} {ast.node} {tokens.EOF}'
                sample.ast_text += f'\n{ast_text}'

                interpreter = basic.Interpreter()
                context = basic.Context('<program>')
                context.symbol_table = symbol_table
                res = interpreter.visit(ast.node, context)
                symbol_table = context.symbol_table
                if res.error:
                    print(f'Interpretation is invalid!: {res.error.as_string()}')
                    invalid_count += 1
                    is_valid = False
                    break

                lex_encoded = data.encode(lexer_text, {})
                ast_encoded = data.encode(ast_text, {})
                if len(lex_encoded) <= block_size and len(ast_encoded) <= block_size:
                    sample.x_in.append(data.add_pad_tokens_and_trim(lex_encoded, block_size))
                    sample.x_out.append(data.add_pad_tokens_and_trim(ast_encoded, block_size))
                    sample.symbols |= symbol_table.symbols
                    if res.value:
                        sample.symbols['_output_list'].append(res.value)
                else:
                    print('Encodings are too long...')
                    invalid_count += 1
                    is_valid = False
                    break
            except Exception:
                invalid_count += 1
                is_valid = False
                break

        if is_valid:
            rows.append(sample.__dict__)

    samples_df = pd.DataFrame(rows)
    # Random shuffle with random seed
    samples_df = samples_df.sample(frac=1, random_state=random.randint(0, 2**31 - 1)).reset_index(drop=True)
    samples_df.to_pickle(dataset_name)  # Save dataset as a Pandas DataFrame (pickled)

    valid_count = len(samples_df)
    print(f"\nValid: {valid_count}, Invalid: {invalid_count}, Func calls: {func_call_count}, Success rate: {valid_count/num_samples*100:.1f}%")


if __name__ == '__main__':
    generate()
