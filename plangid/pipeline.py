import os.path
import os
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

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
                                CountVectorizer(
                                    tokenizer=tokenize,
                                    decode_error="ignore",
                                    analyzer="word",
                                    lowercase=False,
                                    ngram_range=(1, 2),
                                    min_df=3,
                                    max_df=0.99,
                                    binary=True,
                                ),
                                "filename",
                            ),
                            (
                                "content",
                                CountVectorizer(
                                    tokenizer=tokenize,
                                    decode_error="ignore",
                                    analyzer="word",
                                    lowercase=False,
                                    ngram_range=(1, 3),
                                    min_df=3,
                                    max_df=0.99,
                                    binary=True,
                                ),
                                "content",
                            ),
                        ],
                    ),
                ),
                (
                    "classifier",
                    ExtraTreesClassifier(
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
                    ),
                ),
            ]
        )

    def fit(self, X, y=None):
        X, y = self._prepare_fit(X, y)
        return super(LanguagePipeline, self).fit(X, y)

    def fit_predict(self, X, y=None):
        X, y = self._prepare_fit(X, y)
        return super(LanguagePipeline, self).fit(X, y)

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
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

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

    @property
    def _content_vectorizer(self) -> CountVectorizer:
        return self.named_steps["features"].named_transformers_["content"]

    @property
    def _classifier(self) -> ExtraTreesClassifier:
        return self.named_steps["classifier"]


def _sum_dicts(a, b):
    r = dict(a)
    for k, v in b.items():
        if k not in r:
            r[k] = v
        else:
            r[k] = r[k] + v
    return r
