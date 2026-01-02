import basic


def test_lexing_float_plus_int():
    """Test lexing the input '3.4+2'"""
    tokens, error = basic.run('<stdin>', "3.4+2")

    assert error is None

    # Should produce 3 tokens: FLOAT, PLUS, INT
    assert len(tokens) == 3
    assert tokens[0].type == basic.TT_FLOAT
    assert tokens[0].value == 3.4
    assert tokens[1].type == basic.TT_PLUS
    assert tokens[2].type == basic.TT_INT
    assert tokens[2].value == 2


def test_lexing_float_multiply_float():
    """Test lexing the input '2.5 * 2.5'"""
    tokens, error = basic.run('<stdin>', "2.5 * 2.5")

    assert error is None

    # Should produce 3 tokens: FLOAT, MUL, FLOAT
    assert len(tokens) == 3
    assert tokens[0].type == basic.TT_FLOAT
    assert tokens[0].value == 2.5
    assert tokens[1].type == basic.TT_MUL
    assert tokens[2].type == basic.TT_FLOAT
    assert tokens[2].value == 2.5


def test_lexing_int_plus_int():
    """Test lexing the input '1 + 2'"""
    tokens, error = basic.run('<stdin>', "1 + 2")

    assert error is None

    # Should produce 3 tokens: INT, PLUS, INT
    assert len(tokens) == 3
    assert tokens[0].type == basic.TT_INT
    assert tokens[0].value == 1
    assert tokens[1].type == basic.TT_PLUS
    assert tokens[2].type == basic.TT_INT
    assert tokens[2].value == 2


def test_lexing_illegal_char():
    """Test lexing the input '1 + d' should raise IllegalError"""
    tokens, error = basic.run('<stdin>', "1 + d")

    assert error is not None
    assert isinstance(error, basic.IllegalError)
    assert error.error_name == 'Illegal Character'
    assert tokens == []
