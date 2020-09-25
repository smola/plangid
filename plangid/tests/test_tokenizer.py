import pytest

from plangid.tokenizer import tokenize


@pytest.mark.parametrize(
    ("text", "output"),
    [
        (b"", ["$", "$"]),
        (b"abc", ["$", "Wabc", "$"]),
        (b"abc1", ["$", "Wabc1", "$"]),
        (b"abc_", ["$", "Wabc_", "$"]),
        (b"a_bc1", ["$", "Wa_bc1", "$"]),
        (b"a-1", ["$", "Wa", "P-1", "#", "$"]),
        (b" \n\t\t", ["$", "$", "$"]),
        (b"a.b", ["$", "Wa", "P.1", "Wb", "$"]),
        (b"0xAF", ["$", "#h", "$"]),
    ],
)
def test_tokenize(text, output):
    assert tokenize(text) == output
