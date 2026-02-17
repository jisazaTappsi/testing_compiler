INT = 'INT'
FLOAT = 'FLOAT'
IDENTIFIER = 'IDENTIFIER'
KEYWORD = 'KEYWORD'
PLUS = 'PLUS'
MINUS = 'MINUS'
MUL = 'MUL'
DIV = 'DIV'
POW = 'POW'
LPAREN = 'LPAREN'
RPAREN = 'RPAREN'
EQ = 'EQ'
EOF = 'EOF'  # End of file
SOF = 'SOF'  # Start of file
PAD = 'PAD'  # After EOF
VAR = 'var'

TOKENS = [
    INT,
    FLOAT,
    IDENTIFIER,
    KEYWORD,
    PLUS,
    MINUS,
    MUL,
    DIV,
    POW,
    LPAREN,
    RPAREN,
    EQ,
    EOF,
    SOF,
    PAD,
    VAR,
]

KEYWORDS = [
    VAR
]

TOKEN_BASIC_IDS = {
    256 + idx: ','.join(str(int(i)) for i in t.encode('utf-8')) for idx, t in enumerate(TOKENS)
}
TOKEN_IDS = {name: 256 + idx for idx, name in enumerate(TOKENS)}
TOKEN_BYTES = {
    256 + idx: bytes(int(i) for i in t.encode('utf-8')) for idx, t in enumerate(TOKENS)
}
