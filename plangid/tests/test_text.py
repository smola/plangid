from scipy.sparse import lil_matrix
from plangid.text import FastCountVectorizer


def equal_csr(a, b):
    return (a != b).nnz == 0


def check_cv(cv, input, output, vocab):
    X = cv.fit_transform(input)
    assert vocab == cv.get_feature_names()
    assert equal_csr(X, output)
    assert equal_csr(X, cv.transform(input))


def test_fastcountvectorizer_ngram1():
    cv = FastCountVectorizer(ngram_range=(1, 1))
    check_cv(
        cv, input=["abc"], output=lil_matrix([[1, 1, 1]]).tocsr(), vocab=["a", "b", "c"]
    )
    check_cv(
        cv, input=["cba"], output=lil_matrix([[1, 1, 1]]).tocsr(), vocab=["a", "b", "c"]
    )
    check_cv(
        cv,
        input=[b"cba", b"ade"],
        output=lil_matrix([[1, 1, 1, 0, 0], [1, 0, 0, 1, 1]]).tocsr(),
        vocab=["a", "b", "c", "d", "e"],
    )


def test_fastcountvectorizer_ngram1_2():
    cv = FastCountVectorizer(ngram_range=(1, 2))
    check_cv(
        cv,
        input=["abc"],
        output=lil_matrix([[1, 1, 1, 1, 1]]).tocsr(),
        vocab=["a", "ab", "b", "bc", "c"],
    )
    check_cv(
        cv,
        input=["cba"],
        output=lil_matrix([[1, 1, 1, 1, 1]]).tocsr(),
        vocab=["a", "b", "ba", "c", "cb"],
    )
    check_cv(
        cv,
        input=[b"cba", b"ade"],
        output=lil_matrix(
            [[1, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1, 1]]
        ).tocsr(),
        vocab=["a", "ad", "b", "ba", "c", "cb", "d", "de", "e"],
    )
