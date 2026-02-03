########################
# IMPORTS
########################
from strings_with_arrows import *
import torch
import data
from util import device, block_size, code_model_name
from code_train import CrossAttentionTransformer

########################
# CONSTANTS
########################

DIGITS = '0123456789'
ABC = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

########################
# ERRORS
########################

class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.error_name = error_name
        self.details = details
        self.pos_start = pos_start
        self.pos_end = pos_end

    def as_string(self):
        result = f'{self.error_name}: {self.details}'
        result += f'\nFile: {self.pos_start.fn}, line {self.pos_start.ln + 1}'
        result += f'\n\n{string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)}'
        return result

class IllegalError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Illegal Syntax', details)

class RTError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Runtime error', details)

########################
# POSITION
########################

class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1
        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

########################
# TOKENS
########################
from tokens import *


class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end.copy()

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'


########################
# LEXER
########################


class Lexer:
    def __init__(self,fn, text):
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens = [Token(TT_SOF), ]
        while self.current_char is not None:
            if self.current_char in [' ', '\t']:
                self.advance()
            elif self.current_char in DIGITS+'.':
                tokens.append(self.make_number())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char in ABC:
                tokens.append(Token(TT_ABC, pos_start=self.pos))
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalError(pos_start, self.pos, f'"{char}"')

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in DIGITS+'.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)


########################
# NODES
########################

class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'

class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node} {self.op_tok} {self.right_node})'


class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = self.node.pos_end

    def __repr__(self):
        return f'({self.op_tok} {self.node})'


########################
# PARSER RESULT
########################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error: self.error = res.error
            return res.node
        return res

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self

########################
# PARSER
########################

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok
    
    @staticmethod
    def get_tree_from_string(text):
        """
        Parse a tree string representation and rebuild AST nodes recursively.
        Format examples:
        - (TT_INT:2 TT_MUL TT_INT:2) -> BinOpNode
        - (TT_MINUS TT_INT:5) -> UnaryOpNode
        - TT_INT:3 -> NumberNode
        """
        import re
        
        text = text.strip()
        
        # Helper function to parse a token string (e.g., "TT_INT:2" or "TT_MUL")
        def parse_token(token_str):
            token_str = token_str.strip()
            # Create a dummy position for tokens
            dummy_pos = Position(0, 0, 0, '<string>', '')
            if ':' in token_str:
                parts = token_str.split(':', 1)
                token_type = parts[0]
                # Try to parse value as int, float, or keep as string
                value_str = parts[1]
                try:
                    if '.' in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    value = value_str
                # Create a token with dummy position
                return Token(token_type, value, pos_start=dummy_pos)
            else:
                return Token(token_str, pos_start=dummy_pos)
        
        # Helper function to find matching closing parenthesis
        def find_matching_paren(text, start_idx):
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == '(':
                    depth += 1
                elif text[i] == ')':
                    depth -= 1
                    if depth == 0:
                        return i
            return -1
        
        # Try to parse as a simple token (NumberNode) - no parentheses
        if not text.startswith('('):
            # Check if it's a token pattern (e.g., "TT_INT:2" or "TT_MUL")
            if re.match(r'^TT_[A-Z_]+(:.+)?$', text):
                tok = parse_token(text)
                return NumberNode(tok)
            else:
                # Fallback: try to parse as a token anyway
                tok = parse_token(text)
                return NumberNode(tok)
        
        # Parse as BinOpNode or UnaryOpNode (both start with '(')
        # Extract content inside the outermost parentheses
        end_idx = find_matching_paren(text, 0)
        if end_idx == -1:
            raise ValueError(f"Unmatched parenthesis in: {text}")
        
        content = text[1:end_idx].strip()
        
        # Split content by spaces, but preserve parentheses groups
        tokens = []
        current_token = ""
        paren_depth = 0
        
        i = 0
        while i < len(content):
            char = content[i]
            if char == '(':
                if paren_depth == 0 and current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
                current_token += char
                paren_depth += 1
            elif char == ')':
                current_token += char
                paren_depth -= 1
                if paren_depth == 0:
                    tokens.append(current_token.strip())
                    current_token = ""
            elif char == ' ' and paren_depth == 0:
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
            else:
                current_token += char
            i += 1
        
        if current_token.strip():
            tokens.append(current_token.strip())
        
        tokens = [t for t in tokens if t]  # Remove empty tokens
        
        if len(tokens) == 2:
            # UnaryOpNode: (op node)
            op_tok = parse_token(tokens[0])
            node = Parser.get_tree_from_string(tokens[1])
            return UnaryOpNode(op_tok, node)
        elif len(tokens) == 3:
            # BinOpNode: (left op right)
            left = Parser.get_tree_from_string(tokens[0])
            op_tok = parse_token(tokens[1])
            right = Parser.get_tree_from_string(tokens[2])
            return BinOpNode(left, op_tok, right)
        else:
            raise ValueError(f"Unexpected number of tokens: {len(tokens)} in: {content}")

    def parse(self):
        if self.current_tok.type == TT_SOF:
            self.advance()

        res = self.expr()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected '+', '-', '*' or '/' but got {self.current_tok.type}"
            ))
        return res

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        elif tok.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))

        elif tok.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ')' but got {self.current_tok.type}"
                ))


        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            f'Expected int of float but got "{tok.type}"'
        ))

    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))

    def expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def bin_op(self, func, ops):
        res = ParseResult()
        left = res.register(func())
        if res.error: return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error: return res

            left = BinOpNode(left, op_tok, right)
        return res.success(left)

########################
# RUNTIME RESULT
########################

class RTResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error: self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self

########################
# VALUES
########################

class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def added_to(self, other):
        if isinstance(other, Number):
            return Number((self.value + other.value)), None

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number((self.value - other.value)), None

    def mul_by(self, other):
        if isinstance(other, Number):
            return Number((self.value * other.value)), None

    def div_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(pos_start=self.pos_start,
                                     pos_end=self.pos_end,
                                     details='Division by zero :(')
            return Number((self.value / other.value)), None

    def __repr__(self):
        return str(self.value)


########################
# INTERPRETER
########################

class Interpreter:
    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node)

    def no_visit_method(self, node):
        raise Exception(f'No visit_{type(node).__name__} defined')

    def visit_NumberNode(self, node):
        return RTResult().success(
            Number(node.tok.value).set_pos(node.pos_start, node.pos_end)
        )

    def visit_BinOpNode(self, node):
        res = RTResult()
        left = res.register(self.visit(node.left_node))
        if res.error: return res
        right = res.register(self.visit(node.right_node))
        if res.error: return res

        error = None
        if node.op_tok.type == TT_PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == TT_MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TT_MUL:
            result, error = left.mul_by(right)
        elif node.op_tok.type == TT_DIV:
            result, error = left.div_by(right)

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))


    def visit_UnaryOpNode(self, node):
        res = RTResult()
        number = res.register(self.visit(node.node))
        if res.error: return res

        error = None
        if node.op_tok.type == TT_MINUS:
            number, error = number.mul_by(Number(-1))
        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))

########################
# RUN
########################

def run(fn, text):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error: return None, error
    print(tokens)

    # Generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error
    print(ast.node)

    # Run
    interpreter = Interpreter()
    res = interpreter.visit(ast.node)

    return res.value, res.error


def inference(token_list):
    """
    Given lexer `token_list` (as produced by `Lexer.make_tokens`),
    use the trained code transformer to predict an AST and
    return the corresponding AST node.
    """

    # Recreate the lexer string exactly as in `data_generator.generate`
    lex_text = ' '.join(t.__repr__() for t in token_list)

    # 2) Load BPE merges and special tokens
    in_merges, out_merges = data.get_merges()

    # 3) Encode lexer string exactly like `get_code_pairs` does
    lex_encoded = data.encode(lex_text, in_merges)
    lex_encoded = data.add_pad_tokens_and_trim(lex_encoded, block_size)

    # 4) Build model input tensors
    x_in = torch.tensor([lex_encoded], dtype=torch.long, device=device)
    context = torch.tensor([[TOKEN_IDS[TT_SOF]]], dtype=torch.long, device=device)

    # 5) Load trained model and run generation
    model = CrossAttentionTransformer().to(device)
    try:
        model.load_state_dict(torch.load(code_model_name, map_location=device))
        model.eval()
    except FileNotFoundError:
        # If no model is available, fall back to the classic parser
        parser = Parser(token_list)
        ast = parser.parse()
        return ast.node

    with torch.no_grad():
        generated = model.generate(x_out=context, x_in=x_in, max_new_tokens=block_size)[0].tolist()

    # 6) Decode generated ids back into AST text and rebuild AST node
    predicted_ast_text = data.decode(generated, out_merges)

    try:
        ast_node = Parser.get_tree_from_string(predicted_ast_text)
    except Exception:
        # If we cannot rebuild the tree, fall back to the standard parser
        parser = Parser(token_list)
        ast = parser.parse()
        return ast.node

    return ast_node


def run_ai(fn, text):
    lexer = Lexer(fn, text)
    token_list, error = lexer.make_tokens()
    if error: return None, error
    print(token_list)

    # Generate AST
    ast_node = inference(token_list)
    print(ast_node)

    # Run
    interpreter = Interpreter()
    res = interpreter.visit(ast_node)

    return res.value, res.error
