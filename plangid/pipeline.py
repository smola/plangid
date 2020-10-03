try:
    import importlib.resources as importlib_resources
except ImportError:
    import importlib_resources
import gzip
import os.path
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier

from .dataset import Dataset
from .tree import explain_path
from .tokenizer import tokenize


__all__ = ["LanguagePipeline"]


class LanguagePipeline(Pipeline):
    def __init__(self):
        super(LanguagePipeline, self).__init__(
            [
                (
                    "features",
                    ColumnTransformer(
                        transformers=[
                            (
                                "filename",
                                self._new_filename_vectorizer(),
                                "filename",
                            ),
                            (
                                "content",
                                self._new_content_vectorizer(),
                                "content",
                            ),
                        ],
                    ),
                ),
                ("classifier", self._new_classifier()),
            ]
        )

    def fit(self, X, y=None):
        X, y = self._prepare_fit(X, y)
        super(LanguagePipeline, self).fit(X, y)
        self._compact()
        return super(LanguagePipeline, self).fit(X, y)

    def fit_predict(self, X, y=None):
        X, y = self._prepare_fit(X, y)
        super(LanguagePipeline, self).fit(X, y)
        self._compact()
        return super(LanguagePipeline, self).fit_predict(X, y)

    def predict(self, X):
        X = self._prepare_predict(X)
        return super(LanguagePipeline, self).predict(X)

    def predict_proba(self, X):
        X = self._prepare_predict(X)
        return super(LanguagePipeline, self).predict_proba(X)

    def _prepare_fit(self, X, y=None):
        if isinstance(X, Dataset):
            X = X.to_df()
        if y is None:
            y = X["class_name"]
        # passing extra columns can make feature selectors fail
        # if dataframes passed to fit and predict have different
        # number of columns
        X = X[["filename", "content"]]
        return X, y

    def _prepare_predict(self, X, y=None):
        if isinstance(X, str):
            sample = Dataset.load_sample(X)
            X = pd.DataFrame(data=[sample])
        X = X[["filename", "content"]]
        return X

    def save(self, path):
        dir = os.path.dirname(path)
        if dir:
            os.makedirs(dir, exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=None):
        if path is None:
            file = importlib_resources.open_binary("plangid", "model.pickle.gz")
        else:
            file = path
        with gzip.open(file, "rb") as f:
            return pickle.load(f)

    def __getstate__(self):
        filename_vocabulary = list(sorted(self._filename_vectorizer.vocabulary_.keys()))
        content_vocabulary = list(sorted(self._content_vectorizer.vocabulary_.keys()))
        return {
            "filename_vocabulary": filename_vocabulary,
            "content_vocabulary": content_vocabulary,
            "classifier": self._classifier,
        }

    def __setstate__(self, d):
        self.__init__()
        filename_vectorizer = self._new_filename_vectorizer(d["filename_vocabulary"])
        content_vectorizer = self._new_content_vectorizer(d["content_vocabulary"])
        feature_union = self.named_steps["features"]
        feature_union.transformers_ = [
            ("filename", filename_vectorizer, "filename"),
            ("content", content_vectorizer, "content"),
        ]
        feature_union._feature_names_in = ["filename", "content"]
        feature_union._columns = ["filename", "content"]
        feature_union._n_features = 2
        feature_union.sparse_output_ = True
        feature_union._remainder = ("remainder", "drop", None)
        self._classifier = d["classifier"]

    def explain(self, file):
        clf = self._classifier
        features = self.get_feature_names()

        sample = Dataset.load_sample(file)
        X = pd.DataFrame(data=[sample])
        sample_feat = self.transform(X).todense().tolist()[0]

        rules = {}
        result = []
        for tree in clf.estimators_:
            got_gt = False
            got_lte = False
            result = None
            rule = []
            for node in explain_path(tree, sample_feat):
                if node[0] is None:
                    classes = {}
                    for class_idx, prob in enumerate(node[2]):
                        if prob > 0:
                            classes[clf.classes_[class_idx]] = prob
                    result = classes
                else:
                    if node[1] == "lte":
                        if not got_lte and not got_gt:
                            rule.append("...")
                            got_lte = True
                            continue
                        elif not got_gt:
                            continue
                    if node[2] < 1:
                        if node[1] == "lte":
                            rule.append("NOT")
                        rule.append(features[node[0]])
                    else:
                        rule.append(features[node[0]])
                        rule.append("<=" if node[1] == "lte" else ">")
                        rule.append("%.1f" % node[2])
            rule = " ".join(rule)
            if rule not in rules:
                rules[rule] = (1, result)
            else:
                n, r = rules[rule]
                rules[rule] = (n + 1, _sum_dicts(r, result))

        result = []
        for rule, data in reversed(
            sorted(rules.items(), key=lambda x: max(x[1][1].values()))
        ):
            classes = {
                k: v for k, v in reversed(sorted(data[1].items(), key=lambda x: x[1]))
            }
            result.append("%s -> %s (%d trees)" % (rule, str(classes), data[0]))

        return "\n".join(result)

    @property
    def _filename_vectorizer(self) -> CountVectorizer:
        return self.named_steps["features"].named_transformers_["filename"]

    @_filename_vectorizer.setter
    def _filename_vectorizer(self, v: CountVectorizer) -> None:
        self.named_steps["features"].named_transformers_["filename"] = v

    def _new_filename_vectorizer(self, vocab=None) -> CountVectorizer:
        if isinstance(vocab, list):
            vocab = {w: i for i, w in enumerate(vocab)}
        return CountVectorizer(
            tokenizer=tokenize,
            decode_error="ignore",
            analyzer="word",
            lowercase=False,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.99,
            binary=True,
            vocabulary=vocab,
        )

    @property
    def _content_vectorizer(self) -> CountVectorizer:
        return self.named_steps["features"].named_transformers_["content"]

    @_content_vectorizer.setter
    def _content_vectorizer(self, v: CountVectorizer) -> None:
        self.named_steps["features"].named_transformers_["content"] = v

    def _new_content_vectorizer(self, vocab=None) -> CountVectorizer:
        if isinstance(vocab, list):
            vocab = {w: i for i, w in enumerate(vocab)}
        return CountVectorizer(
            tokenizer=tokenize,
            decode_error="ignore",
            analyzer="word",
            lowercase=False,
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.99,
            binary=True,
            vocabulary=vocab,
        )

    @property
    def _classifier(self) -> ExtraTreesClassifier:
        return self.named_steps["classifier"]

    @_classifier.setter
    def _classifier(self, clf: ExtraTreesClassifier) -> None:
        # XXX: named_steps is a property without setter
        self.steps[-1] = ("classifier", clf)

    def _new_classifier(self) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_split=2,
            min_samples_leaf=1,
            min_impurity_decrease=0.0,
            max_depth=None,
            max_features="sqrt",
            max_samples=0.3,
            ccp_alpha=0.002,
            bootstrap=True,
            class_weight="balanced",
            n_jobs=-1,
        )

    def _compact(self) -> None:
        clf = self._classifier
        # non-zero importance feature indices
        feat_indices = np.argwhere(clf.feature_importances_)

        filename_vocab = self._filename_vectorizer.vocabulary_
        filename_vocab_size = len(filename_vocab)
        filename_feat_indices = feat_indices[feat_indices < filename_vocab_size]
        filename_words = [
            w for w, i in sorted(filename_vocab.items()) if i in filename_feat_indices
        ]
        filename_vocab = {w: i for i, w in enumerate(filename_words)}
        self._filename_vectorizer = self._new_filename_vectorizer(filename_vocab)

        content_vocab = self._content_vectorizer.vocabulary_
        content_feat_indices = feat_indices[feat_indices >= filename_vocab_size]
        content_feat_indices -= filename_vocab_size
        content_words = [
            w for w, i in sorted(content_vocab.items()) if i in content_feat_indices
        ]
        content_vocab = {w: i for i, w in enumerate(content_words)}
        self._content_vectorizer = self._new_content_vectorizer(content_vocab)

        self._classifier = self._new_classifier()


def _sum_dicts(a, b):
    r = dict(a)
    for k, v in b.items():
        if k not in r:
            r[k] = v
        else:
            r[k] = r[k] + v
    return r
