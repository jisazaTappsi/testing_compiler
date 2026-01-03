import basic
from basic import TT_EOF


def test_lexing_float_plus_int():
    """Test lexing the input '3.4+2'"""
    lexer = basic.Lexer('<stdin>', "3.4+2")
    tokens, error = lexer.make_tokens()

    assert error is None

    # Should produce 3 tokens: FLOAT, PLUS, INT, EOF
    assert len(tokens) == 4
    assert tokens[0].type == basic.TT_FLOAT
    assert tokens[0].value == 3.4
    assert tokens[1].type == basic.TT_PLUS
    assert tokens[2].type == basic.TT_INT
    assert tokens[2].value == 2
    assert tokens[3].type == TT_EOF


def test_lexing_float_multiply_float():
    """Test lexing the input '2.5 * 2.5'"""
    lexer = basic.Lexer('<stdin>', "2.5 * 2.5")
    tokens, error = lexer.make_tokens()

    assert error is None

    # Should produce 3 tokens: FLOAT, MUL, FLOAT, EOF
    assert len(tokens) == 4
    assert tokens[0].type == basic.TT_FLOAT
    assert tokens[0].value == 2.5
    assert tokens[1].type == basic.TT_MUL
    assert tokens[2].type == basic.TT_FLOAT
    assert tokens[2].value == 2.5
    assert tokens[3].type == TT_EOF


def test_lexing_int_plus_int():
    """Test lexing the input '1 + 2'"""
    lexer = basic.Lexer('<stdin>', "1 + 2")
    tokens, error = lexer.make_tokens()

    assert error is None

    # Should produce 3 tokens: INT, PLUS, INT, EOF
    assert len(tokens) == 4
    assert tokens[0].type == basic.TT_INT
    assert tokens[0].value == 1
    assert tokens[1].type == basic.TT_PLUS
    assert tokens[2].type == basic.TT_INT
    assert tokens[2].value == 2
    assert tokens[3].type == TT_EOF


def test_lexing_illegal_char():
    """Test lexing the input '1 + d' should raise IllegalError"""
    lexer = basic.Lexer('<stdin>', "1 + d")
    tokens, error = lexer.make_tokens()

    assert error is not None
    assert isinstance(error, basic.IllegalError)
    assert error.error_name == 'Illegal Character'
    assert tokens == []


def test_parsing_syntax_error_missing_operand():
    """Test parsing with a syntax error: missing operand after operator"""
    ast, error = basic.run('<stdin>', "1 +")
    
    assert error is not None
    assert isinstance(error, basic.InvalidSyntaxError)
    assert error.error_name == 'Illegal Syntax'
    assert ast is None


def test_parsing_comprehensive_valid_ast():
    """Test parsing a complex expression that covers all main features:
    - Integers and floats
    - All operators: +, -, *, /
    - Operator precedence (multiplication/division before addition/subtraction)
    """
    # Expression: 10 + 2.5 * 3 - 4.2 / 2
    # This tests: addition, multiplication, subtraction, division
    # with both integers and floats, and proper operator precedence
    # Expected AST structure: (10 + (2.5 * 3)) - (4.2 / 2)
    ast, error = basic.run('<stdin>', "10 + 2.5 * 3 - 4.2 / 2")
    
    assert error is None
    assert ast is not None
    
    # Verify it's a BinOpNode (subtraction at the top level due to left associativity)
    assert isinstance(ast, basic.BinOpNode)
    assert ast.op_tok.type == basic.TT_MINUS
    
    # Left side should be addition: 10 + (2.5 * 3)
    assert isinstance(ast.left_node, basic.BinOpNode)
    assert ast.left_node.op_tok.type == basic.TT_PLUS
    
    # Right side should be division: 4.2 / 2
    assert isinstance(ast.right_node, basic.BinOpNode)
    assert ast.right_node.op_tok.type == basic.TT_DIV
    
    # Verify addition's left side: 10 (int)
    assert isinstance(ast.left_node.left_node, basic.NumberNode)
    assert ast.left_node.left_node.tok.type == basic.TT_INT
    assert ast.left_node.left_node.tok.value == 10
    
    # Verify addition's right side is multiplication: 2.5 * 3
    assert isinstance(ast.left_node.right_node, basic.BinOpNode)
    assert ast.left_node.right_node.op_tok.type == basic.TT_MUL
    
    # Verify multiplication operands: 2.5 (float) * 3 (int)
    assert isinstance(ast.left_node.right_node.left_node, basic.NumberNode)
    assert ast.left_node.right_node.left_node.tok.type == basic.TT_FLOAT
    assert ast.left_node.right_node.left_node.tok.value == 2.5
    
    assert isinstance(ast.left_node.right_node.right_node, basic.NumberNode)
    assert ast.left_node.right_node.right_node.tok.type == basic.TT_INT
    assert ast.left_node.right_node.right_node.tok.value == 3
    
    # Verify division operands: 4.2 (float) / 2 (int)
    assert isinstance(ast.right_node.left_node, basic.NumberNode)
    assert ast.right_node.left_node.tok.type == basic.TT_FLOAT
    assert ast.right_node.left_node.tok.value == 4.2
    
    assert isinstance(ast.right_node.right_node, basic.NumberNode)
    assert ast.right_node.right_node.tok.type == basic.TT_INT
    assert ast.right_node.right_node.tok.value == 2
