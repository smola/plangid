import re

_TOK_PATTERN = re.compile(r"(0x[0-9a-fA-F]+|\d+| +|\t+|\w+|(\W)\2*)")
_HEX_PATTERN = re.compile(r"^0x[0-9a-fA-F]+$")

MAX_TOKEN_LENGTH = 10


def tokenize(text):
    if isinstance(text, bytes):
        text = text.decode("latin-1")
    tokens = ["$"]
    for line in text.splitlines():
        for m in re.finditer(_TOK_PATTERN, line):
            token = m.group(0)
            if _HEX_PATTERN.match(token):
                tokens.append("#h")
            elif token.isdigit():
                tokens.append("#")
            elif token[0].isalnum() or token[0] == "_":
                token = token[:MAX_TOKEN_LENGTH]
                tokens.append(f"W{token}")
            elif token[0] in (" ", "\t"):
                continue
            else:
                tokens.append(f"P{token[0]}{len(token)}")
        tokens.append("$")
    if len(tokens) == 1:
        tokens.append("$")
    return tokens
