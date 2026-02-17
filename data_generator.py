import math
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
    #return 0 if random.random() < 0.0 else log_scale_int(1, 2**31, base=10)
    return 0 if random.random() < 0 else log_scale_int(1, 9999)


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
    
    # Randomly add more factors with MUL or DIV operators
    num_ops = random.randint(0, 2)  # 0-2 additional operations
    for _ in range(num_ops):
        op = random.choice(['*', '/'])
        factor = generate_factor(depth, max_depth, allowed_vars)
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
    
    # Randomly add more terms with PLUS or MINUS operators
    num_ops = random.randint(0, 2)  # 0-2 additional operations
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
    max_depth = 3
    expr = generate_expr(depth=0, max_depth=max_depth, allowed_vars=vars_list)

    # If expression is too long, regenerate with lower max_depth
    while len(expr) > max_length//3:
        max_depth = max(1, max_depth - 1)
        expr = generate_expr(depth=0, max_depth=max_depth, allowed_vars=vars_list)

    return expr


# Variable names: letters only, no digits. Exclude names that are keywords.
_VAR_NAME_LETTERS = 'abcdefghijklmnopqrstuwxyz'  # single letters (no 'v' to avoid 'var')


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


def generate_program_statements() -> list:
    """Generates a short program with valid statements. Each statement is either a variable declaration
    ('var x = expr') or a standalone expression. Expressions may use previously declared variables."""
    declared = []
    statements = []
    num_statements = random.randint(2, 5)

    for _ in range(num_statements):
        if not declared or random.random() < 0.6:
            # Variable declaration: var name = expr
            name = _new_var_name(declared)
            expr = generate_arithmetic_expression(use_variables=True, allowed_vars=declared)
            statements.append(f"var {name} = {expr}")
            declared.append(name)
        else:
            # Standalone expression (can use declared variables)
            expr = generate_arithmetic_expression(use_variables=True, allowed_vars=declared)
            statements.append(expr)

    return statements



def introduce_random_error(text):
    """
    Introduce a random error to a string.
    Errors can be: character substitution, deletion, insertion, or swap.
    """
    if len(text) == 0:
        return text

    #error_type = random.choice(['substitute', 'delete', 'insert', 'swap'])
    # IN arithmetic only inserting a non-arithmetic char makes sense, as any other operation would change the
    # meaning of the program itself
    error_type = random.choice(['insert', ])
    text_list = list(text)

    if error_type == 'substitute':
        # Substitute a random character
        idx = random.randint(0, len(text_list) - 1)
        # Replace with a random printable character
        text_list[idx] = random.choice(basic.LETTERS)
    elif error_type == 'delete':
        # Delete a random character
        idx = random.randint(0, len(text_list) - 1)
        text_list.pop(idx)
    elif error_type == 'insert':
        # Insert a random character at a random position
        idx = random.randint(0, len(text_list))
        text_list.insert(idx, random.choice(basic.LETTERS))
    elif error_type == 'swap' and len(text_list) > 1:
        # Swap two adjacent characters
        idx = random.randint(0, len(text_list) - 2)
        text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
    
    return ''.join(text_list)


class Sample:
    lexer_text: str
    ast_text: str
    text: str
    x_in: list
    x_out: list
    symbols: dict = {'_output_list': []}
    id: Optional[int]

    def __init__(self, statements, idx):
        self.lexer_text = ''
        self.ast_text = ''
        self.text = '\n'.join(statements)
        self.x_in = []
        self.x_out = []
        self.symbols['_output_list'] = []
        self.id = idx


def generate():
    invalid_count = 0
    rows = []
    texts = set()

    for idx in range(num_samples):
        if idx % 1_000 == 0:
            print(f"loaded: {(idx/num_samples)*100:.2f}%")
        #text = generate_arithmetic_expression()
        is_valid = True
        statements = generate_program_statements()
        symbol_table = basic.get_symbol_table()
        sample = Sample(statements, idx)

        for text in statements:
            if text in texts:
                invalid_count += 1
                is_valid = False
                break
            texts.add(text)

            # Verify it can be lexed and parsed
            lexer = basic.Lexer('<stdin>', text)
            try:
                token_list, error = lexer.make_tokens()
                if error:
                    print('Lexing is invalid!')
                    invalid_count += 1
                    is_valid = False
                    break
                lexer_text = ' '.join(t.__repr__() for t in token_list)
                sample.lexer_text += f'\n{lexer_text}'

                # Try to parse
                parser = basic.Parser(token_list)
                ast = parser.parse()
                if ast.error:
                    print('Parsing is invalid!')
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
                    invalid_count += 1
                    is_valid = False
                    break
            except Exception as e:
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
    print(f"\nValid: {valid_count}, Invalid: {invalid_count}, Success rate: {valid_count/num_samples*100:.1f}%")


if __name__ == '__main__':
    generate()
