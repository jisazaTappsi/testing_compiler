TT_INT = 'TT_INT'
TT_FLOAT = 'TT_FLOAT'
TT_PLUS = 'TT_PLUS'
TT_MINUS = 'TT_MINUS'
TT_MUL = 'TT_MUL'
TT_DIV = 'TT_DIV'
TT_LPAREN = 'TT_LPAREN'
TT_RPAREN = 'TT_RPAREN'
TT_EOF = 'EOF'  # End of file
TT_SOF = 'SOF'  # Start of file
TT_PAD = 'PAD'  # After EOF
TT_ABC = 'TT_ABC'

TOKENS = [
    TT_INT,
    TT_FLOAT,
    TT_PLUS,
    TT_MINUS,
    TT_MUL,
    TT_DIV,
    TT_LPAREN,
    TT_RPAREN,
    TT_EOF,
    TT_SOF,
    TT_PAD,
    TT_ABC
]

TOKEN_BASIC_IDS = {
    256 + idx: ','.join(str(int(i)) for i in t.encode('utf-8')) for idx, t in enumerate(TOKENS)
}
TOKEN_IDS = {name: 256 + idx for idx, name in enumerate(TOKENS)}
TOKEN_BYTES = {
    256 + idx: bytes(int(i) for i in t.encode('utf-8')) for idx, t in enumerate(TOKENS)
}
