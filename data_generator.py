import csv
import math
import random

import basic
import tokens
from util import *

num_samples = 1_000_000


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

    return 0 if random.random() < 0.0 else log_scale_int(1, 9999)


def generate_number():
    """Generate a random integer or float (log scale: smaller numbers more likely)."""
    # Include 0 with small probability; otherwise log-scale in [1, 9999]
    

    if random.random() < 0.3:  # 30% chance of float
        integer_part = int_part()
        decimal_part = random.randint(0, 99)  # 2 digits id drawn from uniform distribution
        return f"{integer_part}.{decimal_part:02d}"
    else:
        return str(int_part())


def generate_factor(depth=0, max_depth=5):
    """
    Generate a factor according to grammar:
    factor : INT|FLOAT
          : (PLUS|MINUS) factor
          : LPAREN expr RPAREN
    """
    if depth >= max_depth:
        # At max depth, only generate numbers to avoid infinite recursion
        return generate_number()
    
    choice = random.random()
    
    if choice < 0.4:  # 40% chance: number
        return generate_number()
    elif choice < 0.7:  # 30% chance: unary operator
        op = random.choice(['+', '-'])
        factor = generate_factor(depth + 1, max_depth)
        return f"{op}{factor}"
    else:  # 30% chance: parentheses
        expr = generate_expr(depth + 1, max_depth)
        return f"({expr})"


def generate_term(depth=0, max_depth=5):
    """
    Generate a term according to grammar:
    term : factor ((MUL|DIV) factor)*
    """
    result = generate_factor(depth, max_depth)
    
    # Randomly add more factors with MUL or DIV operators
    num_ops = random.randint(0, 2)  # 0-2 additional operations
    for _ in range(num_ops):
        op = random.choice(['*', '/'])
        factor = generate_factor(depth, max_depth)
        result = f"{result}{op}{factor}"
    
    return result


def generate_expr(depth=0, max_depth=5):
    """
    Generate an expression according to grammar:
    expr : (term(PLUS|MINUS) term)*
    """
    if depth >= max_depth:
        # At max depth, generate a simple term
        return generate_term(depth, max_depth)
    
    result = generate_term(depth, max_depth)
    
    # Randomly add more terms with PLUS or MINUS operators
    num_ops = random.randint(0, 2)  # 0-2 additional operations
    for _ in range(num_ops):
        op = random.choice(['+', '-'])
        term = generate_term(depth, max_depth)
        result = f"{result}{op}{term}"
    
    return result


def generate_valid_expression():
    """
    Generate a valid arithmetic expression that can be parsed.
    Optionally limit the length of the generated expression.
    """
    max_length = block_size

    # Start with a reasonable max_depth based on block_size, Deeper expressions tend to be longer
    max_depth = min(5, max_length // 10)
    expr = generate_expr(depth=0, max_depth=max_depth)

    # If expression is too long, regenerate with lower max_depth
    while len(expr) > max_length:
        max_depth = max(1, max_depth - 1)
        expr = generate_expr(depth=0, max_depth=max_depth)
    
    return expr


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
        text_list[idx] = random.choice(basic.ABC)
    elif error_type == 'delete':
        # Delete a random character
        idx = random.randint(0, len(text_list) - 1)
        text_list.pop(idx)
    elif error_type == 'insert':
        # Insert a random character at a random position
        idx = random.randint(0, len(text_list))
        text_list.insert(idx, random.choice(basic.ABC))
    elif error_type == 'swap' and len(text_list) > 1:
        # Swap two adjacent characters
        idx = random.randint(0, len(text_list) - 2)
        text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
    
    return ''.join(text_list)


def generate():
    """Generate valid arithmetic expressions."""
    valid_count = 0
    invalid_count = 0
    dataset = []
    lexer_texts = set()

    for idx in range(num_samples):
        if idx % 1_000 == 0:
            print(f"loaded: {(idx/num_samples)*100:.2f}%")
        # Generate a valid expression
        text = generate_valid_expression()
        text_error = text

        # Introduce random error with 50% probability
        has_error = introduce_error and random.random() < 0.5
        if has_error:
            text_error = introduce_random_error(text)

        # Verify it can be lexed and parsed
        lexer = basic.Lexer('<stdin>', text)
        lexer_error = basic.Lexer('<stdin>', text_error)
        try:
            token_list, error = lexer.make_tokens()
            if error:
                print('Lexing is invalid!')
                invalid_count += 1
                continue
            lexer_text = ' '.join(t.__repr__() for t in token_list)

            tokens_error, error = lexer_error.make_tokens()
            if error:
                print('Lexing is invalid!')
                invalid_count += 1
                continue
            lexer_text_error = ' '.join(t.__repr__() for t in tokens_error)

            if lexer_text in lexer_texts:
                print('Lexer is duplicated, will continue...')
                continue

            # Try to parse
            parser = basic.Parser(token_list)
            ast = parser.parse()
            if ast.error:
                print('Parsing is invalid!')
                invalid_count += 1
                continue

            interpreter = basic.Interpreter()
            res = interpreter.visit(ast.node, basic.Context('<program>'))
            if res.error:
                print(f'Interpretation is invalid!: {res.error.as_string()}')
                invalid_count += 1
                continue

            valid_count += 1
            dataset.append((lexer_text_error, f'{tokens.TT_SOF} {ast.node} {tokens.TT_EOF}', res.value, has_error, idx))
        except Exception as e:
            invalid_count += 1
            continue

    # Write dataset to CSV file once at the end
    with open(dataset_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataset)

    print(f"\nValid: {valid_count}, Invalid: {invalid_count}, Success rate: {valid_count/num_samples*100:.1f}%")


if __name__ == '__main__':
    generate()
