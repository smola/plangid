from .dataset import Dataset
from .tree import explain_path

import pandas as pd
import numpy as np
import os.path
import os
import pickle


class LanguagePipeline:
    def __init__(
        self,
        ngram_range=(1, 5),
        min_df=4,
        max_df=0.99,
        n_estimators=200,
        min_samples_split=4,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.pipeline = self._create_pipeline()

    def fit(self, dataset):
        if isinstance(dataset, Dataset):
            df = dataset.to_df()
        else:
            df = dataset
        self.pipeline.fit(df, df["class_name"])

    def predict(self, X):
        if isinstance(X, str):
            sample = Dataset.load_sample(X)
            X = pd.DataFrame(data=[sample])
            return self.pipeline.predict(X)[0]
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

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
        fe = self.pipeline.named_steps["feature_extraction"]
        clf = self.pipeline.named_steps["clf"]

        sample = Dataset.load_sample(file)
        X = pd.DataFrame(data=[sample])
        sample_feat = fe.transform(X).todense().tolist()[0]

        rules = {}
        result = []
        for tree in clf.estimators_:
            s = []
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
                        rule.append(self._feature_name(node[0]))
                    else:
                        rule.append(self._feature_name(node[0]))
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

    def _feature_name(self, feature_idx):
        fe = self.pipeline.named_steps["feature_extraction"]
        cv_filename = fe.transformer_list[0][1].named_steps["vectorize"]
        cv_content = fe.transformer_list[1][1].named_steps["vectorize"]
        first_len = len(cv_filename.vocabulary_)
        if feature_idx < first_len:
            return "filename '%s'" % self._vocab_idx_to_name(
                cv_filename.vocabulary_, feature_idx
            )
        return "content '%s'" % self._vocab_idx_to_name(
            cv_content.vocabulary_, feature_idx - first_len
        )

    def _vocab_idx_to_name(self, vocab, idx):
        for n, i in vocab.items():
            if i == idx:
                if isinstance(n, bytes):
                    n = n.decode("latin-1")
                return n
        return None

    def _create_pipeline(self):
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            [
                ("feature_extraction", self._create_feature_extraction_pipeline()),
                ("clf", self._create_classifier()),
            ],
            verbose=True,
        )

        return pipeline

    def _create_classifier(self):
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",
            n_jobs=-1,
        )

    def _create_feature_extraction_pipeline(self):
        from sklearn.pipeline import FeatureUnion

        return FeatureUnion(
            transformer_list=[
                ("filename", self._create_filename_feature_extraction_pipeline()),
                ("content", self._create_content_feature_extraction_pipeline()),
            ]
        )

    def _create_content_feature_extraction_pipeline(self):
        from .text import FastCountVectorizer

        vectorizer = FastCountVectorizer(
            ngram_range=self.ngram_range, min_df=self.min_df, max_df=self.max_df,
        )

        # from sklearn.feature_extraction.text import TfidfTransformer

        # tfidf = TfidfTransformer(sublinear_tf=True, use_idf=False)

        from sklearn.pipeline import Pipeline
        from .utils import ItemSelector

        return Pipeline(
            [
                ("select", ItemSelector("content")),
                ("vectorize", vectorizer),
                # ("tfidf", tfidf,),
            ]
        )

    def _create_filename_feature_extraction_pipeline(self):
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(
            input="content",
            encoding="utf-8",
            decode_error="replace",
            analyzer="word",
            lowercase=False,
            min_df=4,
            max_df=1.0,
            dtype=np.uint32,
        )

        # from sklearn.feature_extraction.text import TfidfTransformer

        # tfidf = TfidfTransformer(sublinear_tf=True, use_idf=False)

        from sklearn.pipeline import Pipeline
        from .utils import ItemSelector

        return Pipeline(
            [
                ("select", ItemSelector("filename")),
                ("vectorize", vectorizer),
                # ("tfidf", tfidf,),
            ]
        )


def _sum_dicts(a, b):
    r = dict(a)
    for k, v in b.items():
        if k not in r:
            r[k] = v
        else:
            r[k] = r[k] + v
    return r
