########################
# IMPORTS
########################
import re
import data
import torch
import string

from strings_with_arrows import *
from util import device, block_size, code_model_name
from code_train import CrossAttentionTransformer

########################
# CONSTANTS
########################

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

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

class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Expected Character', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Illegal Syntax', details)

class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'Runtime error', details)
        self.context = context

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context
        while ctx:
            result += f'    File {pos.fn} line {str(pos.ln+1)}, in {ctx.display_name}\n'
            pos = ctx.parent_entry_pos
            ctx = ctx.parent
        return f'Traceback (most recent call last):\n{result}'

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name}: {self.details}'
        result += f'\n\n{string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)}'
        return result

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

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

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
        token_list = [Token(SOF), ]
        while self.current_char is not None:
            if self.current_char in [' ', '\t']:
                self.advance()
            elif self.current_char in DIGITS + '.':
                token_list.append(self.make_number())
            elif self.current_char in LETTERS:
                token_list.append(self.make_identifier())
            elif self.current_char == '+':
                token_list.append(Token(PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                token_list.append(Token(MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '*':
                token_list.append(Token(MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                token_list.append(Token(DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                token_list.append(Token(LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                token_list.append(Token(RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '!':
                tok, error = self.make_not_equals()
                if error: return [], error
                token_list.append(tok)
            elif self.current_char == '=':
                token_list.append(self.make_equals())
            elif self.current_char == '<':
                token_list.append(self.make_less_than())
            elif self.current_char == '>':
                token_list.append(self.make_greater_than())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalError(pos_start, self.pos, f'"{char}"')

        token_list.append(Token(EOF, pos_start=self.pos))
        return token_list, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(INT, int(num_str), pos_start, self.pos)
        else:
            return Token(FLOAT, float(num_str), pos_start, self.pos)

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in LETTERS_DIGITS:
            id_str += self.current_char
            self.advance()

        tok_type = KEYWORD if id_str in KEYWORDS else IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)

    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            return Token(NE, pos_start=pos_start, pos_end=self.pos), None
        
        self.advance()
        return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")

    def make_equals(self):
        tok_type = EQ
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            tok_type = EE
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_less_than(self):
        tok_type = LT
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            tok_type = LTE
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        tok_type = GT
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            tok_type = GTE
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

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


class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end

    def __repr__(self):
        return f'{self.var_name_tok}'


class VarAssignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end

    def __repr__(self):
        return f'{self.var_name_tok}:{self.value_node}'


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


class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = self.else_case or self.cases[len(self.cases)-1][0].pos_end


class ForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.body_node.pos_end


class WhileNode:
    def __init__(self, condition_node, body_node):
        self.condition_node = condition_node
        self.body_node = body_node

        self.pos_start = self.condition_node.pos_start
        self.pos_end = self.body_node.pos_end


########################
# PARSER RESULT
########################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0

    def register_advance(self):
        self.advance_count += 1

    def register(self, res):
        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.advance_count == 0:  # haven't advanced since
            self.error = error
        return self

########################
# PARSER
########################

class Parser:
    def __init__(self, token_list):
        self.token_list = token_list
        self.tok_idx = -1
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.token_list):
            self.current_tok = self.token_list[self.tok_idx]
        return self.current_tok
    
    @staticmethod
    def get_tree_from_string(text):
        """
        Parse a tree string representation and rebuild AST nodes recursively.
        Format examples:
        - (INT:2 MUL INT:2) -> BinOpNode
        - (MINUS INT:5) -> UnaryOpNode
        - INT:3 -> NumberNode
        - IDENTIFIER:x -> VarAccessNode (variable retrieval)
        - IDENTIFIER:x:INT:5 -> VarAssignNode (assignment; value can be any expr string)
        - IDENTIFIER:x:(INT:2 ADD INT:3) -> VarAssignNode with expr value
        """

        text = text.strip()
        
        # Helper function to parse a token string (e.g., "INT:2" or "MUL")
        def parse_token(token_str):
            token_str = token_str.strip()
            # Create a dummy position for token_list
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
        
        # Try to parse as a simple token or identifier (no parentheses)
        if not text.startswith('('):
            # Variable: IDENTIFIER:name (access) or IDENTIFIER:name:expr (assignment)
            if text.startswith('IDENTIFIER:'):
                parts = text.split(':', 2)  # at most 3 parts so value can contain colons
                if len(parts) == 2:
                    # VarAccessNode: IDENTIFIER:name_var
                    tok = parse_token(f"IDENTIFIER:{parts[1]}")
                    return VarAccessNode(tok)
                elif len(parts) == 3:
                    # VarAssignNode: IDENTIFIER:name_var:value (value is expr string)
                    tok = parse_token(f"IDENTIFIER:{parts[1]}")
                    value_node = Parser.get_tree_from_string(parts[2].strip())
                    return VarAssignNode(tok, value_node)
                else:
                    raise ValueError(f"Invalid IDENTIFIER format: {text}")
            # Number or other token (e.g. "INT:2", "FLOAT:3.14", "MUL")
            if re.match(r'^[A-Z_]+(:.+)?$', text):
                tok = parse_token(text)
                return NumberNode(tok)
            else:
                tok = parse_token(text)
                return NumberNode(tok)
        
        # Parse as BinOpNode or UnaryOpNode (both start with '(')
        # Extract content inside the outermost parentheses
        end_idx = find_matching_paren(text, 0)
        if end_idx == -1:
            raise ValueError(f"Unmatched parenthesis in: {text}")
        
        content = text[1:end_idx].strip()
        
        # Split content by spaces, but preserve parentheses groups
        token_list = []
        current_token = ""
        paren_depth = 0
        
        i = 0
        while i < len(content):
            char = content[i]
            if char == '(':
                if paren_depth == 0 and current_token.strip():
                    token_list.append(current_token.strip())
                    current_token = ""
                current_token += char
                paren_depth += 1
            elif char == ')':
                current_token += char
                paren_depth -= 1
                if paren_depth == 0:
                    token_list.append(current_token.strip())
                    current_token = ""
            elif char == ' ' and paren_depth == 0:
                if current_token.strip():
                    token_list.append(current_token.strip())
                    current_token = ""
            else:
                current_token += char
            i += 1
        
        if current_token.strip():
            token_list.append(current_token.strip())
        
        token_list = [t for t in token_list if t]  # Remove empty token_list

        if len(token_list) == 1:
            return Parser.get_tree_from_string(token_list[0])
        elif len(token_list) == 2:
            # UnaryOpNode: (op node)
            op_tok = parse_token(token_list[0])
            node = Parser.get_tree_from_string(token_list[1])
            return UnaryOpNode(op_tok, node)
        elif len(token_list) == 3:
            # BinOpNode: (left op right)
            left = Parser.get_tree_from_string(token_list[0])
            op_tok = parse_token(token_list[1])
            right = Parser.get_tree_from_string(token_list[2])
            return BinOpNode(left, op_tok, right)
        else:
            raise ValueError(f"Unexpected number of tokens: {len(token_list)} in: {content}")

    def parse(self):
        if self.current_tok.type == SOF:
            self.advance()

        res = self.expr()
        if not res.error and self.current_tok.type != EOF:
            return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected '+', '-', '*' or '/' but got {self.current_tok.type}"
            ))
        return res

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (PLUS, MINUS):
            res.register_advance()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        elif tok.type in (INT, FLOAT):
            res.register_advance()
            self.advance()
            return res.success(NumberNode(tok))

        elif tok.type == IDENTIFIER:
            res.register_advance()
            self.advance()
            return res.success(VarAccessNode(tok))

        elif tok.type == LPAREN:
            res.register_advance()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == RPAREN:
                res.register_advance()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ')' but got {self.current_tok.type}"
                ))

        elif tok.matches(KEYWORD, IF):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)

        elif tok.matches(KEYWORD, FOR):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)

        elif tok.matches(KEYWORD, WHILE):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)

        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            f'Expected int, float, "+", "-", "(" or identifier but got "{tok.type}"'
        ))

    def if_expr(self):
        res = ParseResult()
        cases = []
        else_case = None

        if not self.current_tok.matches(KEYWORD, IF):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{IF}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(KEYWORD, THEN):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{THEN}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        expr = res.register(self.expr())
        if res.error: return res
        cases.append((condition, expr))

        while self.current_tok.matches(KEYWORD, ELIF):
            res.register_advance()
            self.advance()

            condition = res.register(self.expr())
            if res.error: return res

            if not self.current_tok.matches(KEYWORD, THEN):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected '{THEN}' but got {self.current_tok.type}"
                ))

            res.register_advance()
            self.advance()

            expr = res.register(self.expr())
            if res.error: return res
            cases.append((condition, expr))

        if self.current_tok.matches(KEYWORD, ELSE):
            res.register_advance()
            self.advance()

            else_case = res.register(self.expr())
            if res.error: return res

        return res.success(IfNode(cases, else_case))

    def for_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(KEYWORD, FOR):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{FOR}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        if self.current_tok.type != IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected identifier but got {self.current_tok.type}"
            ))

        var_name = self.current_tok
        res.register_advance()
        self.advance()

        if self.current_tok.type != EQ:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '=' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        start_value = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(KEYWORD, TO):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{TO}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        end_value = res.register(self.expr())
        if res.error: return res

        if self.current_tok.matches(KEYWORD, STEP):
            res.register_advance()
            self.advance()

            step_value = res.register(self.expr())
            if res.error: return res
        else:
            step_value = None

        if not self.current_tok.matches(KEYWORD, THEN):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{THEN}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        body = res.register(self.expr())
        if res.error: return res

        return res.success(ForNode(var_name_tok=var_name,
                start_value_node=start_value,
                end_value_node=end_value,
                step_value_node=step_value,
                body_node=body))

    def while_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(KEYWORD, WHILE):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{WHILE}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(KEYWORD, THEN):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{THEN}' but got {self.current_tok.type}"
            ))

        res.register_advance()
        self.advance()

        body = res.register(self.expr())
        if res.error: return res

        return res.success(WhileNode(condition, body))

    def term(self):
        return self.bin_op(self.factor, (MUL, DIV))

    def arith_expr(self):
        return self.bin_op(self.term, (PLUS, MINUS))

    def comp_expr(self):
        res = ParseResult()
        if self.current_tok.matches(KEYWORD, NOT):
            op_tok = self.current_tok
            res.register_advance()
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op_tok, node))

        node = res.register(self.bin_op(self.arith_expr, (EE, NE, LT, GT, LTE, GTE)))
        if res.error:
            return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f'Expected int, float, identifier "+", "-", "(", or "{NOT}" but got "{self.current_tok.type}"'
            ))
        return res.success(node)

    def expr(self):
        res = ParseResult()
        if self.current_tok.matches(KEYWORD, VAR):
            res.register_advance()
            self.advance()
            if self.current_tok.type != IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                        pos_start=self.current_tok.pos_start, pos_end=self.current_tok.pos_end,
                        details=f'In assignment was looking for IDENTIFIER but got {self.current_tok.type}'
                    )
                )
            var_name = self.current_tok

            res.register_advance()
            self.advance()
            if self.current_tok.type != EQ:
                return res.failure(InvalidSyntaxError(
                        pos_start=self.current_tok.pos_start, pos_end=self.current_tok.pos_end,
                        details=f'In assignment was looking for EQ but got {self.current_tok.type}'
                    )
                )
            res.register_advance()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))

        node = res.register(self.bin_op(self.comp_expr, ((KEYWORD, AND), (KEYWORD, OR))))
        if res.error:
            return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f'Expected var, int, float, "+", "-", "(" or identifier but got "{self.current_tok.type}"'
            ))
        return res.success(node)

    def bin_op(self, func, ops):
        res = ParseResult()
        left = res.register(func())
        if res.error: return res

        while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
            op_tok = self.current_tok
            res.register_advance()
            self.advance()
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
        self.set_context()

    def set_context(self, context=None):
        self.context = context
        return self

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def added_to(self, other):
        if isinstance(other, Number):
            return Number((self.value + other.value)).set_context(self.context), None

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number((self.value - other.value)).set_context(self.context), None

    def mul_by(self, other):
        if isinstance(other, Number):
            return Number((self.value * other.value)).set_context(self.context), None

    def div_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(pos_start=self.pos_start,
                                     pos_end=self.pos_end,
                                     details='Division by zero :(',
                                     context=self.context)
            return Number((self.value / other.value)).set_context(self.context), None

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None

    def notted(self):
        return Number(int(self.value == 0)).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return str(self.value)

    def is_true(self):
        return self.value != 0


########################
# CONTEXT
########################

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None


########################
# SYMBOL TABLE
########################

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None

    def get(self, name):
        value = self.symbols.get(name)
        if value is None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]


########################
# INTERPRETER
########################

class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} defined')

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)

        if value is None:
            return res.failure(RTError(
                pos_start=node.pos_start,
                pos_end=node.pos_end,
                details=f'Variable "{var_name}" is not defined',
                context=context
            ))

        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.error: return res
        context.symbol_table.set(var_name, value)
        return res.success(value)

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error: return res
        right = res.register(self.visit(node.right_node, context))
        if res.error: return res

        error = None
        if node.op_tok.type == PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == MUL:
            result, error = left.mul_by(right)
        elif node.op_tok.type == DIV:
            result, error = left.div_by(right)
        elif node.op_tok.type == EE:
            result, error = left.get_comparison_eq(right)
        elif node.op_tok.type == NE:
            result, error = left.get_comparison_ne(right)
        elif node.op_tok.type == LT:
            result, error = left.get_comparison_lt(right)
        elif node.op_tok.type == GT:
            result, error = left.get_comparison_gt(right)
        elif node.op_tok.type == LTE:
            result, error = left.get_comparison_lte(right)
        elif node.op_tok.type == GTE:
            result, error = left.get_comparison_gte(right)
        elif node.op_tok.matches(KEYWORD, AND):
            result, error = left.anded_by(right)
        elif node.op_tok.matches(KEYWORD, OR):
            result, error = left.ored_by(right)

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res

        error = None
        if node.op_tok.type == MINUS:
            number, error = number.mul_by(Number(-1))
        elif node.op_tok.matches(KEYWORD, NOT):
            number, error = number.notted()

        if error:
            return res.failure(error)

        return res.success(number.set_pos(node.pos_start, node.pos_end))

    def visit_IfNode(self, node, context):
        res = RTResult()

        for condition, expr in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.error: return res

            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.error: return res
                return res.success(expr_value)

        if node.else_case:
            else_value = res.register(self.visit(node.else_case, context))
            if res.error: return res
            return res.success(else_value)

        return res.success(None)

    def visit_ForNode(self, node, context):
        res = RTResult()

        start_value = res.register(self.visit(node.start_value_node, context))
        if res.error: return res

        end_value = res.register(self.visit(node.end_value_node, context))
        if res.error: return res

        if node.step_value_node:
            step_value = res.register(self.visit(node.step_value_node, context))
            if res.error: return res
        else:
            step_value = Number(1)  # default step is 1

        i = start_value.value

        if step_value.value >= 0:
            condition = lambda: i < end_value.value
        else:
            condition = lambda: i > end_value.value

        while condition():
            context.symbol_table.set(node.var_name_tok.value, Number(i))
            i += step_value.value

            res.register(self.visit(node.body_node, context))
            if res.error: return res

        return res.success(None)

    def visit_WhileNode(self, node, context):
        res = RTResult()

        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.error: return res

            if not condition.is_true(): break

            res.register(self.visit(node.body_node, context))
            if res.error: return res

        return res.success(None)


########################
# RUN
########################

def get_symbol_table():
    table = SymbolTable()
    # Start from the same built-ins as the global symbol table (NULL/TRUE/FALSE).
    table.symbols.update(global_symbol_table.symbols)
    table.set(NULL, Number(0))
    table.set(TRUE, Number(1))
    table.set(FALSE, Number(0))
    return table


global_symbol_table = SymbolTable()
global_symbol_table.set(NULL, Number(0))
global_symbol_table.set(TRUE, Number(1))
global_symbol_table.set(FALSE, Number(0))


def run(fn, text):
    lexer = Lexer(fn, text)
    token_list, error = lexer.make_tokens()
    if error: return None, error

    # Generate AST
    parser = Parser(token_list)
    ast = parser.parse()
    if ast.error: return None, ast.error

    # Run
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    res = interpreter.visit(ast.node, context)

    print(f'Symbol table: {global_symbol_table.symbols}')
    return res.value, res.error


def inference(token_list):
    """
    Given lexer `token_list` (as produced by `Lexer.make_tokens`),
    use the trained code transformer to predict an AST and
    return the corresponding AST node.
    """
    lex_text = ' '.join(t.__repr__() for t in token_list)

    lex_merges, ast_merges = data.get_merges()
    lex_encoded = data.encode(lex_text, lex_merges)
    lex_encoded = data.add_pad_tokens_and_trim(lex_encoded, block_size)

    model = CrossAttentionTransformer().to(device)
    model.load_state_dict(torch.load(code_model_name, map_location=device))
    model.eval()

    predicted_ast_text = model.inference(lex_encoded, ast_merges)

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

    # Generate AST
    parser = Parser(token_list)
    ast = parser.parse()
    if ast.error:
        print(ast.error.as_string())
        print('using AI')
        ast_node = inference(token_list)
    else:
        ast_node = ast.node

    # Run
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    res = interpreter.visit(ast_node, context)

    return res.value, res.error
