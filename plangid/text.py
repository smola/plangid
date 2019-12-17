from plangid.dataset import Dataset

from array import array
from collections import defaultdict
import numbers
import numpy as np
from operator import itemgetter
import scipy.sparse as sp


class FastCountVectorizer:
    """
    TODO: Add note about CountVecotirzer whitespace
          normalization, which we avoid here.
    """

    def __init__(self, ngram_range, min_df=1, max_df=1.0):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary_ = None

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        vocab, X = self._count_vocab(raw_documents)
        X, _ = self._limit_features(X, vocab)
        X = self._sort_features(X, vocab)
        self.vocabulary_ = vocab
        return X

    def transform(self, raw_documents):
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)
        return X

    def get_feature_names(self):
        return [
            self._to_string(t)
            for t, i in sorted(self.vocabulary_.items(), key=itemgetter(1))
        ]

    def _to_string(self, s):
        if isinstance(s, bytes):
            return s.decode("latin-1")
        return s

    def _count_vocab(self, raw_documents, fixed_vocab=False):
        if fixed_vocab:
            vocab = self.vocabulary_
        else:
            vocab = defaultdict()
            vocab.default_factory = vocab.__len__

        values = array("i")
        j_indices = array("i")
        # j_indices = []
        indptr = [0]

        for doc in raw_documents:
            counters = defaultdict(int)
            for term in self._analyze(doc):
                if not fixed_vocab or term in vocab:
                    idx = vocab[term]
                    counters[idx] += 1
            j_indices.extend(counters.keys())
            values.extend(counters.values())
            indptr.append(len(j_indices))

        # TODO: add 32bit warning

        # j_indices = np.asarray(j_indices, dtype=np.int64)
        j_indices = np.frombuffer(j_indices, dtype=np.intc)
        indptr = np.asarray(indptr, dtype=np.int64)
        values = np.frombuffer(values, dtype=np.intc)

        # remove default dict behavior
        if not fixed_vocab:
            vocab = dict(vocab)

        X = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(vocab)),
            dtype=np.int32,
        )

        X.sort_indices()

        return vocab, X

    def _sort_features(self, X, vocabulary):
        """Sort features by name
        Returns a reordered matrix and modifies the vocabulary in place
        """
        sorted_features = sorted(vocabulary.items())
        map_index = np.empty(len(sorted_features), dtype=X.indices.dtype)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode="clip")
        return X

    def _limit_features(self, X, vocabulary):
        """Remove too rare or too common features.
        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.
        This does not prune samples with zero features.
        """
        min_df = self.min_df
        max_df = self.max_df
        needs_limit = (
            (isinstance(min_df, numbers.Integral) and min_df > 1) or min_df > 0.0
        ) or ((isinstance(min_df, numbers.Integral) and max_df > 0) or max_df < 1.0)

        if not needs_limit:
            return X, set()

        n_doc = X.shape[0]
        high = max_df if isinstance(max_df, numbers.Integral) else max_df * n_doc
        low = min_df if isinstance(min_df, numbers.Integral) else min_df * n_doc
        if high < low:
            raise ValueError("max_df corresponds to < documents than min_df")

        # Calculate a mask based on document frequencies
        dfs = self._document_frequency(X)
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for term, old_index in list(vocabulary.items()):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError(
                "After pruning, no terms remain. Try a lower"
                " min_df or a higher max_df."
            )
        return X[:, kept_indices], removed_terms

    def _document_frequency(self, X):
        """Count the number of non-zero values for each feature in sparse X."""
        if sp.isspmatrix_csr(X):
            return np.bincount(X.indices, minlength=X.shape[1])
        else:
            return np.diff(X.indptr)

    def _analyze(self, doc):
        doc_len = len(doc)
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(0, doc_len - n + 1):
                yield doc[i : i + n]
